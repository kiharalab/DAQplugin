# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
DAQ Score computation pipeline for ChimeraX.

This module provides the core computation pipeline for DAQ scores,
using ChimeraX's native volume handling and ONNX Runtime for inference.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Callable
from time import perf_counter

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


def normalize_npy_output_path(output_path: Union[str, Path]) -> Path:
    """
    Return the path that np.save will actually create for an NPY output.

    numpy appends ".npy" when the path does not already end with that suffix.
    Normalize before saving so logging and downstream loading use the real file.
    """
    output_path = Path(output_path)
    if str(output_path).lower().endswith(".npy"):
        return output_path
    return Path(f"{output_path}.npy")


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

    # Save the original volume's id before resampling
    original_vol_id = vol.id_string

    # Resample to 1 Angstrom grid spacing
    run(session, f"volume resample #{vol.id_string} spacing 1")

    # Show the original volume (resample command hides it by default)
    run(session, f"show #{original_vol_id}")

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
    # Flatten and filter positive values in one pass
    flat = vol_data.ravel()
    dmax = float(flat.max())
    if dmax <= 0.0:
        return 0.0

    tic = dmax / nbins
    if tic <= 0.0:
        return 0.0

    # Compute histogram directly on flat array (faster than filtering first)
    counts, _ = np.histogram(flat[flat > 0.0], bins=nbins, range=(0.0, dmax))

    # Log-transform counts (vectorized)
    with np.errstate(divide='ignore'):
        log_counts = np.where(counts > 0, np.log(counts), 0.0)

    total_sum = log_counts.sum()
    if total_sum <= 0.0:
        return 0.0

    sum_cut = total_sum * c

    # Find cutoff bin using cumsum (vectorized)
    cumsum = np.cumsum(log_counts)
    cutoff_indices = np.where(cumsum >= sum_cut)[0]
    cutoff_bin = cutoff_indices[0] if len(cutoff_indices) > 0 else 0

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
    # Work with float32 to save memory
    vol = np.clip(vol_data, 0, None).astype(np.float32, copy=False)

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
) -> Tuple[np.ndarray, np.ndarray]:
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
    tuple
        (points, density) where points is (N, 3) in world coordinates (X, Y, Z)
        and density is (N,) raw density values at each point
    """
    # Apply stride
    if stride > 1:
        vol_s = vol_data[::stride, ::stride, ::stride]
        mask = vol_s >= contour
        idx_zyx = np.argwhere(mask)
        if idx_zyx.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        idx_zyx = idx_zyx * stride
    else:
        mask = vol_data >= contour
        idx_zyx = np.argwhere(mask)
        if idx_zyx.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Get density values at selected voxels (from raw data)
    density = vol_data[idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]].astype(np.float32)

    # Downsample if needed
    if max_points is not None and idx_zyx.shape[0] > max_points:
        sel = np.random.choice(idx_zyx.shape[0], size=max_points, replace=False)
        idx_zyx = idx_zyx[sel]
        density = density[sel]

    # Convert ZYX indices to XYZ world coordinates
    idx_xyz = idx_zyx[:, ::-1].astype(np.float32)  # ZYX -> XYZ

    # Convert to world coordinates: origin + idx * step
    origin_xyz = np.array(origin, dtype=np.float32)
    step_xyz = np.array(step, dtype=np.float32)

    points = origin_xyz + idx_xyz * step_xyz

    return points.astype(np.float32), density


# Global cache for numba-compiled function
_numba_extract_fn = None


def _get_numba_extract_fn():
    """Get or create the numba-compiled extraction function."""
    global _numba_extract_fn
    if _numba_extract_fn is not None:
        return _numba_extract_fn

    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _extract(padded, centers, patches, r):
        N = centers.shape[0]
        ps = 2 * r + 1
        for i in prange(N):
            cz, cy, cx = centers[i]
            for dz in range(ps):
                for dy in range(ps):
                    for dx in range(ps):
                        patches[i, dz, dy, dx] = padded[cz + dz, cy + dy, cx + dx]

    _numba_extract_fn = _extract
    return _numba_extract_fn


def _extract_patches_numpy(padded, centers, patches, r):
    """Pure numpy patch extraction (fallback)."""
    N = centers.shape[0]
    ps = 2 * r + 1
    for i in range(N):
        cz, cy, cx = centers[i]
        patches[i] = padded[cz:cz+ps, cy:cy+ps, cx:cx+ps]


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

    Uses numba for parallel extraction if available, otherwise falls back
    to optimized numpy loop.

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

    # Convert to ZYX for indexing and round to integers
    voxel_centers = np.round(voxel_idx_xyz[:, ::-1]).astype(np.int32)

    # Pad volume to handle boundary cases (pad with zeros)
    padded = np.pad(vol_data, pad_width=r, mode='constant', constant_values=0)

    # Centers point to top-left corner of patch in padded volume
    # (voxel_centers is already the center in original volume,
    #  adding r from padding makes it the center in padded volume,
    #  then we use it directly since we iterate from 0 to patch_size)
    centers = voxel_centers + r - r  # This equals voxel_centers, but conceptually: center_in_padded - r = corner
    # Actually: center in padded = voxel_centers + r (due to padding)
    # Corner = center - r = voxel_centers + r - r = voxel_centers
    # But we need to clip to valid range
    Dz, Dy, Dx = vol_data.shape
    centers[:, 0] = np.clip(voxel_centers[:, 0], 0, Dz - 1)
    centers[:, 1] = np.clip(voxel_centers[:, 1], 0, Dy - 1)
    centers[:, 2] = np.clip(voxel_centers[:, 2], 0, Dx - 1)

    # Pre-allocate output
    patches = np.zeros((N, patch_size, patch_size, patch_size), dtype=np.float32)

    # Try numba first (much faster, parallel), fall back to numpy
    padded_f32 = padded.astype(np.float32)
    try:
        extract_fn = _get_numba_extract_fn()
        extract_fn(padded_f32, centers, patches, r)
    except (ImportError, Exception):
        # Numba not available or failed, use numpy fallback
        _extract_patches_numpy(padded_f32, centers, patches, r)

    # Swap XZ axes for model
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
    density: np.ndarray,
    ref_contour: float,
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
    density : np.ndarray
        Raw density values at each point (N,)
    ref_contour : float
        Contour threshold for reference distribution filtering

    Returns
    -------
    np.ndarray
        Combined scores array (N, 32): [xyz(3), aa_log(20), atom_log(6), ss_log(3)]
    """
    eps = 1e-12

    # Reference mask: points with density >= ref_contour
    ref_mask = density >= ref_contour

    if not np.any(ref_mask):
        # Fallback: use all points if no points pass the threshold
        ref_mask = np.ones(len(density), dtype=bool)

    # Compute reference distributions from filtered points
    ref_aa = np.clip(aa_probs[ref_mask].mean(axis=0), eps, 1.0)
    ref_atom = np.clip(atom_probs[ref_mask].mean(axis=0), eps, 1.0)
    ref_ss = np.clip(ss_probs[ref_mask].mean(axis=0), eps, 1.0)

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
    batch_size: int = 0,
    max_points: int = 500000,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Optional[Path], dict]:
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
        Batch size for inference (0 = auto)
    max_points : int
        Maximum number of points (default: 500000)
    model_path : str, optional
        Path to ONNX model (uses bundled model if None)
    progress_callback : callable, optional
        Progress callback function(current, total, message)
    gpu_id : int
        NVIDIA device id for tensorrt/cuda backends.
    backend : str
        Inference backend: "auto" (platform chain), "tensorrt", "cuda",
        "directml", "mlx", "mlx-cpu", or "cpu". See onnx_model.load_model.

    Returns
    -------
    tuple
        (points, scores, actual_output_path) where scores is (N, 32) array
        and actual_output_path is the Path where the file was saved (or None if not saved)
    """
    import time

    timings = {
        "input_data_processing": 0.0,
        "daq_computing": 0.0,
        "score_assignment": 0.0,
    }

    def update_progress(current, total, msg=""):
        if progress_callback:
            progress_callback(current, total, msg)
        else:
            session.logger.status(f"{msg} ({current}/{total})")

    # Pipeline start measures steps 1-4 (resample + extract + patches) as
    # the user-visible "input_data_processing" bucket. Each step also
    # records its own sub-key for fine-grained timing logs.
    pipeline_start = perf_counter()

    # Step 1: Unify and resample volume
    update_progress(0, 6, "Unifying and resampling volume...")
    t0 = perf_counter()

    # Unify map first if needed
    if isinstance(map_input, (str, Path)):
        map_input_unified = unify_map_if_needed(str(map_input))
    else:
        map_input_unified = map_input

    # Track volumes before resampling to detect if a new one is created
    from chimerax.map import Volume
    volumes_before = set(m for m in session.models.list() if isinstance(m, Volume))

    # Then resample
    vol = resize_map_to_1a(session, map_input_unified)
    timings['1_resample'] = time.perf_counter() - t0

    # Check if a new volume was created (resampling happened)
    volumes_after = set(m for m in session.models.list() if isinstance(m, Volume))
    new_volumes = volumes_after - volumes_before

    # Step 2: Get volume data
    update_progress(1, 6, "Extracting volume data...")
    t0 = time.perf_counter()
    data = vol.data.matrix().copy()  # (Z, Y, X) numpy array - copy to detach from volume
    origin = vol.data.origin  # (x, y, z)
    step = vol.data.step  # Should be ~(1, 1, 1) after resample

    # Close only newly created resampled volume(s) to clean up ChimeraX GUI
    # This keeps the original volume the user loaded/selected
    for new_vol in new_volumes:
        # Only delete if it's the volume we used AND has "resample" in name (safety check)
        if new_vol is vol and "resample" in new_vol.name.lower():
            try:
                new_vol.delete()
                session.logger.info("Cleaned up resampled volume from GUI")
            except Exception:
                pass  # Volume may already be closed or not deletable

    # Normalize volume
    data_norm = normalize_volume(data)
    timings['2_volume_data'] = time.perf_counter() - t0

    # Step 3: Extract points above contour * 0.5 (more points for better coverage)
    # Reference distribution will be filtered by original contour later
    extraction_contour = contour * 0.5
    update_progress(2, 6, "Extracting grid points...")
    t0 = time.perf_counter()
    points, density = extract_threshold_points(data, origin, step, contour=extraction_contour, stride=stride, max_points=max_points)
    timings['3_extract_points'] = time.perf_counter() - t0

    n_points = points.shape[0]
    session.logger.info(f"Extracted {n_points} points above contour {extraction_contour} (extraction threshold)")
    session.logger.info(f"Reference will use points with density >= {contour} (original contour)")

    if n_points == 0:
        session.logger.warning("No points found above contour threshold!")
        timings["input_data_processing"] = perf_counter() - pipeline_start
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 32), dtype=np.float32),
                None, timings)

    # Step 4: Extract patches
    update_progress(3, 6, f"Extracting {n_points} patches...")
    t0 = time.perf_counter()
    patches = extract_patches(data_norm, points, origin, step, patch_size=11)
    timings['4_extract_patches'] = time.perf_counter() - t0

    # Steps 1-4 done; record the aggregate input-processing bucket.
    timings["input_data_processing"] = perf_counter() - pipeline_start

    # Step 5: Run ONNX inference
    update_progress(4, 6, "Loading model and running inference...")
    t2 = perf_counter()
    model = load_model(model_path, backend=backend, gpu_id=gpu_id)

    session.logger.info(f"Running inference on {n_points} patches...")

    def inference_progress(current, total):
        update_progress(4, 6, f"Inference: {current}/{total} patches")

    t0 = time.perf_counter()
    aa_probs, atom_probs, ss_probs = model.predict_batched(patches, batch_size=batch_size, progress_callback=inference_progress)
    t3 = perf_counter()
    timings["daq_computing"] = t3 - t2

    # Step 6: Compute log-ratio scores (reference filtered by original contour)
    update_progress(5, 6, "Computing DAQ scores...")
    t4 = perf_counter()
    ref_points = np.sum(density >= contour)
    session.logger.info(f"Reference points: {ref_points}/{n_points} (density >= {contour})")
    scores = compute_log_ratio_scores(points, aa_probs, atom_probs, ss_probs, density, contour)
    t5 = perf_counter()
    timings["score_assignment"] = t5 - t4

    # Save results if output path provided
    actual_output_path = None
    if output_path:
        output_path = normalize_npy_output_path(output_path)
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
            actual_output_path = normalize_npy_output_path(safe_dir / output_path.name)
            np.save(str(actual_output_path), scores)
            session.logger.warning(f"Could not write to {output_path.parent}, saving to {actual_output_path}")
            session.logger.info(f"Saved DAQ scores to: {actual_output_path}")
            session.logger.info(f"Output shape: {scores.shape}")

    update_progress(6, 6, "Done!")

    return points, scores, actual_output_path, timings


def get_heavy_atom_coords(structure) -> np.ndarray:
    """
    Extract heavy atom (non-H) coordinates from a ChimeraX structure.

    Parameters
    ----------
    structure : chimerax.atomic.Structure
        ChimeraX structure model

    Returns
    -------
    np.ndarray
        Heavy atom coordinates (N, 3) in Angstroms
    """
    atoms = structure.atoms
    # Filter out hydrogen atoms
    heavy_mask = atoms.elements.names != 'H'
    heavy_atoms = atoms[heavy_mask]
    coords = heavy_atoms.scene_coords  # Use scene coordinates for alignment
    return coords.astype(np.float32)


def compute_log_ratio_scores_pdb(
    points: np.ndarray,
    aa_probs: np.ndarray,
    atom_probs: np.ndarray,
    ss_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute DAQ log-ratio scores for PDB version.
    Reference distributions are computed from ALL points (all heavy atoms).

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

    # Compute reference distributions from ALL points (all heavy atoms are valid)
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


def compute_daq_scores_pdb(
    session,
    map_input,
    structure,
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 0,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Optional[Path], dict]:
    """
    Compute DAQ scores for PDB structure (heavy atom positions).

    This version extracts patches at heavy atom coordinates from the structure
    instead of grid points from the map.

    Parameters
    ----------
    session : chimerax.core.session.Session
        ChimeraX session
    map_input : str, Path, or chimerax.map.Volume
        Path to input MRC/MAP file OR a ChimeraX Volume object
    structure : chimerax.atomic.Structure
        Structure model whose heavy atom coordinates will be used
    output_path : str or Path, optional
        Path to save output NPY file
    batch_size : int
        Batch size for inference (0 = auto)
    model_path : str, optional
        Path to ONNX model (uses bundled model if None)
    progress_callback : callable, optional
        Progress callback function(current, total, message)
    gpu_id : int
        NVIDIA device id for tensorrt/cuda backends.
    backend : str
        Inference backend: "auto" (platform chain), "tensorrt", "cuda",
        "directml", "mlx", "mlx-cpu", or "cpu". See onnx_model.load_model.

    Returns
    -------
    tuple
        (points, scores, actual_output_path) where scores is (N, 32) array
        and actual_output_path is the Path where the file was saved (or None if not saved)
    """
    import time

    timings = {
        "input_data_processing": 0.0,
        "daq_computing": 0.0,
        "score_assignment": 0.0,
    }

    def update_progress(current, total, msg=""):
        if progress_callback:
            progress_callback(current, total, msg)
        else:
            session.logger.status(f"{msg} ({current}/{total})")

    # Pipeline start covers steps 1-4 (resample + extract + patches).
    pipeline_start = perf_counter()

    # Step 1: Unify and resample volume
    update_progress(0, 6, "Unifying and resampling volume...")
    t0 = perf_counter()

    # Unify map first if needed
    if isinstance(map_input, (str, Path)):
        map_input_unified = unify_map_if_needed(str(map_input))
    else:
        map_input_unified = map_input

    # Track volumes before resampling to detect if a new one is created
    from chimerax.map import Volume
    volumes_before = set(m for m in session.models.list() if isinstance(m, Volume))

    # Then resample
    vol = resize_map_to_1a(session, map_input_unified)
    timings['1_resample'] = time.perf_counter() - t0

    # Check if a new volume was created (resampling happened)
    volumes_after = set(m for m in session.models.list() if isinstance(m, Volume))
    new_volumes = volumes_after - volumes_before

    # Step 2: Get volume data
    update_progress(1, 6, "Extracting volume data...")
    t0 = time.perf_counter()
    data = vol.data.matrix().copy()  # (Z, Y, X) numpy array - copy to detach from volume
    origin = vol.data.origin  # (x, y, z)
    step = vol.data.step  # Should be ~(1, 1, 1) after resample

    # Close only newly created resampled volume(s) to clean up ChimeraX GUI
    # This keeps the original volume the user loaded/selected
    for new_vol in new_volumes:
        # Only delete if it's the volume we used AND has "resample" in name (safety check)
        if new_vol is vol and "resample" in new_vol.name.lower():
            try:
                new_vol.delete()
                session.logger.info("Cleaned up resampled volume from GUI")
            except Exception:
                pass  # Volume may already be closed or not deletable

    # Normalize volume
    data_norm = normalize_volume(data)
    timings['2_volume_data'] = time.perf_counter() - t0

    # Step 3: Extract heavy atom coordinates from structure
    update_progress(2, 6, "Extracting heavy atom coordinates...")
    t0 = time.perf_counter()
    points = get_heavy_atom_coords(structure)
    timings['3_extract_coords'] = time.perf_counter() - t0

    n_points = points.shape[0]
    session.logger.info(f"Extracted {n_points} heavy atom coordinates from structure")

    if n_points == 0:
        session.logger.warning("No heavy atoms found in structure!")
        timings["input_data_processing"] = perf_counter() - pipeline_start
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 32), dtype=np.float32),
                None, timings)

    # Step 4: Extract patches at heavy atom positions
    update_progress(3, 6, f"Extracting {n_points} patches...")
    t0 = time.perf_counter()
    patches = extract_patches(data_norm, points, origin, step, patch_size=11)
    timings['4_extract_patches'] = time.perf_counter() - t0

    # Steps 1-4 done; record the aggregate input-processing bucket.
    timings["input_data_processing"] = perf_counter() - pipeline_start

    # Step 5: Run ONNX inference
    update_progress(4, 6, "Loading model and running inference...")
    t2 = perf_counter()
    model = load_model(model_path, backend=backend, gpu_id=gpu_id)

    session.logger.info(f"Running inference on {n_points} patches...")

    def inference_progress(current, total):
        update_progress(4, 6, f"Inference: {current}/{total} patches")

    t0 = time.perf_counter()
    aa_probs, atom_probs, ss_probs = model.predict_batched(patches, batch_size=batch_size, progress_callback=inference_progress)
    t3 = perf_counter()
    timings["daq_computing"] = t3 - t2

    # Step 6: Compute log-ratio scores (PDB version uses all points for reference)
    update_progress(5, 6, "Computing DAQ scores...")
    t4 = perf_counter()
    scores = compute_log_ratio_scores_pdb(points, aa_probs, atom_probs, ss_probs)
    t5 = perf_counter()
    timings["score_assignment"] = t5 - t4

    # Save results if output path provided
    actual_output_path = None
    if output_path:
        output_path = normalize_npy_output_path(output_path)
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
            actual_output_path = normalize_npy_output_path(safe_dir / output_path.name)
            np.save(str(actual_output_path), scores)
            session.logger.warning(f"Could not write to {output_path.parent}, saving to {actual_output_path}")
            session.logger.info(f"Saved DAQ scores to: {actual_output_path}")
            session.logger.info(f"Output shape: {scores.shape}")

    update_progress(6, 6, "Done!")

    return points, scores, actual_output_path, timings
