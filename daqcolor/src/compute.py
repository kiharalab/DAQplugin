# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
DAQ Score computation pipeline for ChimeraX.

This module provides the core computation pipeline for DAQ scores,
using ChimeraX's native volume handling and ONNX Runtime for inference.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

# Handle imports for both ChimeraX plugin and standalone use
try:
    from .onnx_model import DAQOnnxModel, get_model_path, load_model
except ImportError:
    from onnx_model import DAQOnnxModel, get_model_path, load_model

# Try to import unify_map (standalone use)
try:
    from map_util.unify_map import Unify_Map as unify_map_function
except ImportError:
    unify_map_function = None


def unify_map_if_needed(map_path: str, temp_dir: str = None) -> str:
    """
    Unify a map file to standard MRC format.

    Parameters
    ----------
    map_path : str
        Path to the input map file
    temp_dir : str, optional
        Safe temporary directory for unified map (use when input dir is protected)

    Returns
    -------
    str
        Path to the unified map file
    """
    if unify_map_function is None:
        # unify_map not available, return original path
        return map_path

    from pathlib import Path

    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    # Determine safe output directory
    # Use provided temp_dir, or try map's parent, or fall back to temp
    if temp_dir is None:
        try:
            # Try to write to the map's directory
            test_file = map_path.parent / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            temp_dir = str(map_path.parent)
        except (PermissionError, OSError):
            # Use system temp directory as fallback
            import tempfile
            temp_dir = tempfile.gettempdir()

    unified_dir = Path(temp_dir) / "unified_map"
    unified_dir.mkdir(parents=True, exist_ok=True)

    unified_map_path = unified_dir / f"{map_path.stem}_unified.mrc"

    if unified_map_path.exists():
        return str(unified_map_path)

    print(f"Unifying map: {map_path} -> {unified_map_path}")
    unify_map_function(str(map_path), str(unified_map_path))
    return str(unified_map_path)


def resize_map_to_1a(session, map_path_or_volume, close_original: bool = False):
    """
    Resample volume to 1 Angstrom voxel size using ChimeraX native.

    Parameters
    ----------
    session : chimerax.core.session.Session
        ChimeraX session
    map_path_or_volume : str, Path, or Volume
        Path to MRC/MAP file or existing Volume model
    close_original : bool
        If True, close the original volume after resampling

    Returns
    -------
    chimerax.map.Volume
        Resampled volume with ~1 Angstrom voxel size
    """
    from chimerax.core.commands import run
    from chimerax.map import Volume

    # Load volume if path provided
    if isinstance(map_path_or_volume, (str, Path)):
        map_path = Path(map_path_or_volume)
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")

        # Open the map using ChimeraX
        models = run(session, f'open "{map_path}"')
        if not models:
            raise RuntimeError(f"Failed to open map: {map_path}")
        vol = models[0]
    elif isinstance(map_path_or_volume, Volume):
        vol = map_path_or_volume
    else:
        raise TypeError(f"Expected path or Volume, got {type(map_path_or_volume)}")

    # Check current voxel size
    step = vol.data.step  # (x_step, y_step, z_step)

    # Check if resampling is needed (within 1% of 1 Angstrom)
    if all(abs(s - 1.0) < 0.01 for s in step):
        session.logger.info(f"Volume already has 1 Å voxel size: {step}")
        return vol

    session.logger.info(f"Resampling volume from voxel size {step} to 1 Å...")

    # Resample to 1 Angstrom grid spacing
    run(session, f"volume resample #{vol.id_string} spacing 1")

    # Get the resampled volume (created as the most recent model)
    resampled = None
    for m in reversed(session.models.list()):
        if isinstance(m, Volume) and m is not vol:
            resampled = m
            break

    if resampled is None:
        raise RuntimeError("Failed to create resampled volume")

    # Optionally close original to save memory
    if close_original and vol is not resampled:
        vol.delete()

    session.logger.info(f"Resampled to voxel size: {resampled.data.step}")
    return resampled


def find_contour_cutoff(vol_data: np.ndarray, c: float = 0.95, nbins: int = 200) -> float:
    """
    Find contour cutoff using DAQ's FindTopX algorithm.

    Parameters
    ----------
    vol_data : np.ndarray
        3D volume data
    c : float
        Fraction of cumulative log-histogram for cutoff (default: 0.95)
    nbins : int
        Number of histogram bins (default: 200)

    Returns
    -------
    float
        Computed contour cutoff value
    """
    vol = np.asarray(vol_data, dtype=np.float64)

    # Only consider positive density values
    dens = vol[vol > 0.0]
    if dens.size == 0:
        return 0.0

    dmax = float(dens.max())
    if dmax <= 0.0:
        return 0.0

    tic = dmax / nbins
    if tic <= 0.0:
        return 0.0

    # Compute histogram
    counts, edges = np.histogram(dens, bins=nbins, range=(0.0, dmax))

    # Log-transform counts
    log_counts = np.zeros_like(counts, dtype=np.float64)
    mask_pos = counts > 0
    log_counts[mask_pos] = np.log(counts[mask_pos])

    total_sum = float(log_counts[mask_pos].sum())
    if total_sum <= 0.0:
        return 0.0

    sum_cut = total_sum * c

    # Find cutoff bin
    cumsum = 0.0
    cutoff_bin = 0
    for i, lc in enumerate(log_counts):
        if lc > 0.0:
            cumsum += lc
            if cumsum >= sum_cut:
                cutoff_bin = i
                break

    return float(tic * cutoff_bin)


def normalize_volume(vol_data: np.ndarray, p_low: float = None, p_high: float = None) -> np.ndarray:
    """
    Normalize volume data using percentile clipping and min-max scaling.

    Parameters
    ----------
    vol_data : np.ndarray
        3D volume data
    p_low : float, optional
        Low percentile value for clipping
    p_high : float, optional
        High percentile value for clipping

    Returns
    -------
    np.ndarray
        Normalized volume in range [0, 1]
    """
    vol = np.maximum(vol_data, 0)  # Clip negative values

    if p_low is None or p_high is None:
        # Use DAQ's FindTopX algorithm for p_high
        p_high = find_contour_cutoff(vol, c=0.95, nbins=200)
        p_low = 0.0

        # Fallback to percentiles if FindTopX fails
        if p_high <= p_low + 1e-8:
            positive = vol[vol > 0]
            if len(positive) > 0:
                p_low, p_high = np.percentile(positive, [1.0, 99.0])
            else:
                p_low, p_high = 0.0, 1.0

    # Clip and scale to [0, 1]
    vol_clip = np.clip(vol, p_low, p_high)
    vmin, vmax = float(vol_clip.min()), float(vol_clip.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(vol_clip, dtype=np.float32)

    vol_norm = (vol_clip - vmin) / (vmax - vmin + 1e-8)
    return vol_norm.astype(np.float32)


def extract_threshold_points(
    vol_data: np.ndarray,
    origin: Tuple[float, float, float],
    step: Tuple[float, float, float],
    contour: float = 0.0,
    stride: int = 2,
    max_points: Optional[int] = 500000,
) -> np.ndarray:
    """
    Extract grid points above contour threshold.

    Parameters
    ----------
    vol_data : np.ndarray
        3D volume data with shape (Z, Y, X)
    origin : tuple
        Volume origin (x, y, z) in Angstroms
    step : tuple
        Voxel size (x, y, z) in Angstroms
    contour : float
        Contour threshold value
    stride : int
        Stride for point sampling (default: 2)
    max_points : int, optional
        Maximum number of points to return

    Returns
    -------
    np.ndarray
        Points array with shape (N, 3) in world coordinates (X, Y, Z)
    """
    # Apply stride
    if stride > 1:
        vol_s = vol_data[::stride, ::stride, ::stride]
        mask = vol_s >= contour
        idx_zyx = np.argwhere(mask)
        if idx_zyx.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        idx_zyx = idx_zyx * stride
    else:
        mask = vol_data >= contour
        idx_zyx = np.argwhere(mask)
        if idx_zyx.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

    # Downsample if needed
    if max_points is not None and idx_zyx.shape[0] > max_points:
        sel = np.random.choice(idx_zyx.shape[0], size=max_points, replace=False)
        idx_zyx = idx_zyx[sel]

    # Convert ZYX indices to XYZ world coordinates
    idx_xyz = idx_zyx[:, ::-1].astype(np.float32)  # ZYX -> XYZ

    # Convert to world coordinates: origin + idx * step
    origin_xyz = np.array(origin, dtype=np.float32)
    step_xyz = np.array(step, dtype=np.float32)

    points = origin_xyz + idx_xyz * step_xyz

    return points.astype(np.float32)


def extract_patches(
    vol_data: np.ndarray,
    points: np.ndarray,
    origin: Tuple[float, float, float],
    step: Tuple[float, float, float],
    patch_size: int = 11,
    swap_xz: bool = True,
) -> np.ndarray:
    """
    Extract 3D patches centered at each point.

    Parameters
    ----------
    vol_data : np.ndarray
        Normalized 3D volume with shape (Z, Y, X)
    points : np.ndarray
        Point coordinates (N, 3) in XYZ world coordinates
    origin : tuple
        Volume origin (x, y, z)
    step : tuple
        Voxel size (x, y, z)
    patch_size : int
        Size of cubic patch (default: 11)
    swap_xz : bool
        If True, transpose patches from (X,Y,Z) to (Z,Y,X) for model

    Returns
    -------
    np.ndarray
        Patches array with shape (N, 1, D, H, W)
    """
    N = points.shape[0]
    r = patch_size // 2

    origin_xyz = np.array(origin, dtype=np.float32)
    step_xyz = np.array(step, dtype=np.float32)

    # Convert world coordinates to voxel indices (XYZ)
    voxel_idx_xyz = (points - origin_xyz) / step_xyz

    # Convert to ZYX for indexing
    voxel_idx_zyx = voxel_idx_xyz[:, ::-1]

    patches = np.zeros((N, patch_size, patch_size, patch_size), dtype=np.float32)

    Dz, Dy, Dx = vol_data.shape

    for i in range(N):
        cz, cy, cx = np.round(voxel_idx_zyx[i]).astype(int)

        # Patch bounds in volume
        z0, z1 = cz - r, cz + r + 1
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1

        # Valid volume region
        vz0, vz1 = max(0, z0), min(Dz, z1)
        vy0, vy1 = max(0, y0), min(Dy, y1)
        vx0, vx1 = max(0, x0), min(Dx, x1)

        # Corresponding patch region
        pz0, pz1 = vz0 - z0, vz1 - z0
        py0, py1 = vy0 - y0, vy1 - y0
        px0, px1 = vx0 - x0, vx1 - x0

        patches[i, pz0:pz1, py0:py1, px0:px1] = vol_data[vz0:vz1, vy0:vy1, vx0:vx1]

    # Swap XZ axes for model: (N, Z, Y, X) -> (N, X, Y, Z) -> transpose to (N, Z, Y, X)
    # Actually the model expects (N, 1, Z, Y, X) after swapping
    if swap_xz:
        patches = np.transpose(patches, (0, 3, 2, 1))  # (N, X, Y, Z)

    # Add channel dimension
    patches = patches[:, np.newaxis, :, :, :]  # (N, 1, D, H, W)

    return patches


def compute_log_ratio_scores(
    points: np.ndarray,
    aa_probs: np.ndarray,
    atom_probs: np.ndarray,
    ss_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute DAQ log-ratio scores from probability predictions.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates (N, 3)
    aa_probs : np.ndarray
        Amino acid probabilities (N, 20)
    atom_probs : np.ndarray
        Atom type probabilities (N, 6)
    ss_probs : np.ndarray
        Secondary structure probabilities (N, 3)

    Returns
    -------
    np.ndarray
        Combined scores array (N, 32): [xyz(3), aa_log(20), atom_log(6), ss_log(3)]
    """
    eps = 1e-12

    # Compute reference distributions (mean across all points)
    ref_aa = np.clip(aa_probs.mean(axis=0), eps, 1.0)
    ref_atom = np.clip(atom_probs.mean(axis=0), eps, 1.0)
    ref_ss = np.clip(ss_probs.mean(axis=0), eps, 1.0)

    # Compute log-ratio scores
    aa_log = np.log(np.clip(aa_probs, eps, 1.0) / ref_aa[None, :]).astype(np.float32)
    atom_log = np.log(np.clip(atom_probs, eps, 1.0) / ref_atom[None, :]).astype(np.float32)
    ss_log = np.log(np.clip(ss_probs, eps, 1.0) / ref_ss[None, :]).astype(np.float32)

    # Concatenate: [xyz(3), aa(20), atom(6), ss(3)] = 32 columns
    scores = np.concatenate(
        [
            points.astype(np.float32),
            aa_log,
            atom_log,
            ss_log,
        ],
        axis=1,
    )

    return scores


def compute_daq_scores(
    session,
    map_input,
    output_path: Optional[Union[str, Path]] = None,
    contour: float = 0.0,
    stride: int = 2,
    batch_size: int = 512,
    max_points: int = 500000,
    model_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Path]]:
    """
    Full DAQ score computation pipeline.

    Parameters
    ----------
    session : chimerax.core.session.Session
        ChimeraX session
    map_input : str, Path, or chimerax.map.Volume
        Path to input MRC/MAP file OR a ChimeraX Volume object
    output_path : str or Path, optional
        Path to save output NPY file
    contour : float
        Contour threshold (default: 0.0)
    stride : int
        Stride for point sampling (default: 2)
    batch_size : int
        Batch size for inference (default: 512)
    max_points : int
        Maximum number of points (default: 500000)
    model_path : str, optional
        Path to ONNX model (uses bundled model if None)
    progress_callback : callable, optional
        Progress callback function(current, total, message)

    Returns
    -------
    tuple
        (points, scores, actual_output_path) where scores is (N, 32) array
        and actual_output_path is the Path where the file was saved (or None if not saved)
    """

    def update_progress(current, total, msg=""):
        if progress_callback:
            progress_callback(current, total, msg)
        else:
            session.logger.status(f"{msg} ({current}/{total})")

    # Step 1: Unify and resample volume
    update_progress(0, 6, "Unifying and resampling volume...")

    # Unify map first if needed
    if isinstance(map_input, (str, Path)):
        map_input_unified = unify_map_if_needed(str(map_input))
    else:
        map_input_unified = map_input

    # Then resample
    vol = resize_map_to_1a(session, map_input_unified)

    # Step 2: Get volume data
    update_progress(1, 6, "Extracting volume data...")
    data = vol.data.matrix()  # (Z, Y, X) numpy array
    origin = vol.data.origin  # (x, y, z)
    step = vol.data.step  # Should be ~(1, 1, 1) after resample

    # Normalize volume
    data_norm = normalize_volume(data)

    # Step 3: Extract points above contour
    update_progress(2, 6, "Extracting grid points...")
    points = extract_threshold_points(data, origin, step, contour=contour, stride=stride, max_points=max_points)

    n_points = points.shape[0]
    session.logger.info(f"Extracted {n_points} points above contour {contour}")

    if n_points == 0:
        session.logger.warning("No points found above contour threshold!")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 32), dtype=np.float32), None

    # Step 4: Extract patches
    update_progress(3, 6, f"Extracting {n_points} patches...")
    patches = extract_patches(data_norm, points, origin, step, patch_size=11)

    # Step 5: Run ONNX inference
    update_progress(4, 6, "Loading model and running inference...")
    model = load_model(model_path)

    session.logger.info(f"Running inference on {n_points} patches...")

    def inference_progress(current, total):
        update_progress(4, 6, f"Inference: {current}/{total} patches")

    aa_probs, atom_probs, ss_probs = model.predict_batched(patches, batch_size=batch_size, progress_callback=inference_progress)

    # Step 6: Compute log-ratio scores
    update_progress(5, 6, "Computing DAQ scores...")
    scores = compute_log_ratio_scores(points, aa_probs, atom_probs, ss_probs)

    # Save results if output path provided
    actual_output_path = None
    if output_path:
        output_path = Path(output_path)
        try:
            # Try to write to the requested directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output_path), scores)
            actual_output_path = output_path
            session.logger.info(f"Saved DAQ scores to: {output_path}")
            session.logger.info(f"Output shape: {scores.shape}")
        except (PermissionError, OSError):
            # Fall back to user's home directory
            safe_dir = Path.home() / "DAQcolor_output"
            safe_dir.mkdir(parents=True, exist_ok=True)
            actual_output_path = safe_dir / output_path.name
            np.save(str(actual_output_path), scores)
            session.logger.warning(f"Could not write to {output_path.parent}, saving to {actual_output_path}")
            session.logger.info(f"Saved DAQ scores to: {actual_output_path}")
            session.logger.info(f"Output shape: {scores.shape}")

    update_progress(6, 6, "Done!")

    return points, scores, actual_output_path
