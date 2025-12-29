# vim: set expandtab shiftwidth=4 softtabstop=4:
import numpy as np
from chimerax.core.commands import (CmdDesc, StringArg, IntArg, BoolArg,
                                    ColormapArg, FloatArg, ModelArg)


# Session Monitor
_MON = {}  # (session, model.id_string) -> dict
'''
def _residue_coords(residues, atom_name="CA"):
    coords = []
    
    have = residues.atoms[residues.atoms.names == 'CA']
    have_res = set(a.residue for a in have)
    for r in residues:
        a = r.find_atom(atom_name)
        if a is not None:
            coords.append(a.coord)
        else:
            xyz = r.atoms.coords
            coords.append(xyz.mean(axis=0) if len(xyz) else (0,0,0))
    return np.asarray(coords, dtype=np.float32)
'''
def _residue_coords(residues, atom_name="CA", use_scene=True):
    """
    residues の座標を 1 残基 1 点で返す。
    use_scene=True: including transform, scene coordinates
    use_scene=False: original coord
    """
    coords = []
    for r in residues:
        a = r.find_atom(atom_name)
        if a is not None:
            if use_scene:
                coords.append(a.scene_coord)   # ★ ここを coord → scene_coord
            else:
                coords.append(a.coord)
        else:
            ats = r.atoms
            if len(ats):
                if use_scene:
                    xyz = ats.scene_coords     # ★ ここを coords → scene_coords
                else:
                    xyz = ats.coords
                coords.append(xyz.mean(axis=0))
            else:
                coords.append((0, 0, 0))
    return np.asarray(coords, dtype=np.float32)


# 追加: _knn_idx を距離閾値対応 & 距離も返す
def _knn_idx(db_pts, q_pts, k=8, radius=None, chunk=2000):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(db_pts)
        if radius is None:
            dist, idx = tree.query(q_pts, k=k)
        else:
            dist, idx = tree.query(q_pts, k=k, distance_upper_bound=float(radius))
        
        if k == 1:
            dist = dist[:, None]
            idx  = idx[:, None]
        return dist, idx
    except Exception:
        # NumPy fallback 
        Nq = q_pts.shape[0]
        out_idx = np.empty((Nq, k), dtype=np.int32)
        out_dist = np.empty((Nq, k), dtype=np.float32)
        for s in range(0, Nq, chunk):
            e = min(Nq, s+chunk)
            q = q_pts[s:e]
            diff = q[:, None, :] - db_pts[None, :, :]
            d2 = np.einsum('mpc,mpc->mp', diff, diff)
            part = np.argpartition(d2, k-1, axis=1)[:, :k]
            sub = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(sub, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
            dist = np.sqrt(np.take_along_axis(sub, order, axis=1))
            if radius is not None:
                mask = dist > float(radius)
                # ダミー: idx を 0 に、dist を inf にして後段で無視
                idx[mask] = 0
                dist[mask] = np.inf
            out_idx[s:e] = idx
            out_dist[s:e] = dist
        return out_dist, out_idx


def _aggregate(pts, aa, q, k=1, radius=None):
    """
    pts: (N,3)
    aa:  (N,C) 
    q:   (M,3)
    k:   int

    return:
      aa_nn:        (M,C) 
      has_neighbor: (M,)  
    """
    dist, idx = _knn_idx(pts, q, k=k, radius=radius)  # dist:(M,k), idx:(M,k)
    N, C = aa.shape
    M = q.shape[0]

    # index
    valid = (idx >= 0) & (idx < N) & np.isfinite(dist)

    # find closest
    best_pos = np.argmin(dist, axis=1)   # (M,)
    rows = np.arange(M)

    # has neighbor?
    has_neighbor = valid.any(axis=1)     # (M,)
    best_is_valid = has_neighbor & valid[rows, best_pos]

    # invalid entries
    safe_idx = np.zeros(M, dtype=np.int64)
    safe_idx[best_is_valid] = idx[rows, best_pos][best_is_valid]

    # get nn
    aa_nn = aa[safe_idx].copy()  # (M,C)
    aa_nn[~best_is_valid] = 0.0  # fill zero

    return aa_nn, has_neighbor

def _window_average_scal(residues, scal, half_window=9):
    """
    residues: ChimeraX ResidueCollection
    scal:     (R,) residue score
    half_window: n-9 ~ n+9 def:9

    戻り値:
      scal_win: (R,) 
    """
    R = len(residues)
    scal = np.asarray(scal, dtype=np.float32)
    out = np.full(R, np.nan, dtype=np.float32)

    # Make array
    chain_ids = np.array([r.chain_id for r in residues], dtype=object)
    resnums   = np.array([r.number   for r in residues], dtype=int)

    for i in range(R):
        c = chain_ids[i]
        n = resnums[i]

        # compute window
        mask = (chain_ids == c) & (resnums >= n - half_window) & (resnums <= n + half_window)
        vals = scal[mask]

        # ignore NaN
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = vals.mean()

    return out



def _recolor(session, model, npy_path, k, cmap, metric, atom_name, clamp_min, clamp_max, radius=3.0, halfwindow=9):
    AA20 = [
    "ALA","VAL","PHE","PRO","MET","ILE","LEU","ASP","GLU","LYS",
    "ARG","SER","THR","TYR","HIS","CYS","ASN","TRP","GLN","GLY"
    ]

    AA_INDEX = {aa:i for i,aa in enumerate(AA20)}
    ATOM_TYPES6 = ["Other","N","CA","C","O","CB"]  #（index 0..5）

    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 32:
        raise ValueError(f"Expected (N,32) numpy file; got {arr.shape}")
    pts  = arr[:, :3].astype(np.float32)
    aa   = arr[:, 3:23].astype(np.float32)
    atom = arr[:, 23:29]
    #ss3 = arr[:, 29:32]  # SS not use



    residues = model.residues
    if residues is None or len(residues) == 0:
        session.logger.warning("No residues in model.")
        return

    q = _residue_coords(residues, atom_name=atom_name, use_scene=True)  # (M,3)
    aa_mean, has_nbr = _aggregate(pts, aa, q, k=k, radius=radius)
    atom_mean, has_nbr = _aggregate(pts, atom, q, k=k, radius=radius)
    # metric
    if metric == "aa_score":
        # 残基のAAタイプに対応する列だけを抽出
        names = np.array([n.upper() for n in residues.names], dtype=object)  # (R,)
        idx = np.array([AA_INDEX.get(n, -1) for n in names], dtype=int)      # (R,)
        scal = np.full((len(residues),), np.nan, dtype=np.float32)
        valid = idx >= 0
        if np.any(valid):
            rows = np.nonzero(valid)[0]
            scal[rows] = aa_mean[rows, idx[valid]]
        #scal = aa_mean.max(axis=1)
        #for a, b, c,me in zip(scal,names,idx,aa_mean):
        #    print(a,b,c,me)
    elif metric.startswith("aa_conf:"):
        aa3 = metric.split(":",1)[1].upper()
        j = AA20.index(aa3)
        scal = aa_mean[:, j]
    elif metric == "atom_score":
        j = ATOM_TYPES6.index("CA")
        scal = atom_mean[:, j]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    scal = _window_average_scal(residues, scal, half_window=halfwindow)

    # --- Input score into B-factor ---
    ats = residues.atoms

    # NaN / inf を 0.0 に置き換え
    scal_for_b = np.asarray(scal, dtype=np.float32).copy()
    bad = ~np.isfinite(scal_for_b)
    if np.any(bad):
        scal_for_b[bad] = 0.0

    # put b-fac values
    bf_vals = np.repeat(scal_for_b, residues.num_atoms)

    # atoms
    ats.bfactors = bf_vals
    # --- 追加ここまで ---


    from chimerax.core.colors import Colormap, Color
    
    if cmap is None:
        colors_rgbs = np.array([
            [1.0,0.0,0.0,1.0],
            [1.0,1.0,1.0,1.0],
            [0.0,0.0,1.0,1.0]
        ],dtype=np.float32
        )
    cmap = cmap or Colormap(name="daq",colors=colors_rgbs,data_values=[-1.0,0.0,1.0])
    # 入力スコアを −1〜1 にクリップ
    if clamp_min is None or clamp_max is None:
        x = np.clip(scal, -1.0, 1.0)          # scal
    else:
        x = np.clip(scal, float(clamp_min), float(clamp_max))

    # 残基ごとの RGBA
    # Handle NaN values in scores before color interpolation
    x_safe = np.where(np.isfinite(x), x, 0.0)
    res_rgba_f = cmap.interpolated_rgba(x_safe)
    
    # Mark residues without neighbors as green
    green = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    if not np.all(has_nbr):
        res_rgba_f[~has_nbr] = green
    # Also mark NaN scores as green
    nan_mask = ~np.isfinite(x)
    if np.any(nan_mask):
        res_rgba_f[nan_mask] = green

    res_rgba = (np.clip(res_rgba_f, 0, 1) * 255).astype(np.uint8)
    residues.ribbon_colors = res_rgba

    # 原子ごとの RGBA（残基→原子に展開）
    ats = residues.atoms
    vals_atom = np.repeat(x_safe, residues.num_atoms)
    atom_rgba_f = cmap.interpolated_rgba(vals_atom)

    if not np.all(has_nbr):
        atom_mask = np.repeat(~has_nbr, residues.num_atoms)
        atom_rgba_f[atom_mask] = green
    # Also mark NaN scores as green for atoms
    if np.any(nan_mask):
        atom_nan_mask = np.repeat(nan_mask, residues.num_atoms)
        atom_rgba_f[atom_nan_mask] = green

    atom_rgba = np.ascontiguousarray(
        (np.clip(atom_rgba_f, 0.0, 1.0) * 255).astype(np.uint8)
    )
    ats.colors = atom_rgba

    session.logger.status(
        f"daqcolor: colored {len(residues)} residues (k={k}, metric={metric})",
        color="blue"
    )
    

# --------- commands ---------

def daqcolor_apply(session, npy_path, model, *, k=1, colormap=None,
                   metric="aa_score", atom_name="CA",
                   clamp_min=None, clamp_max=None,half_window=9):
    _recolor(session, model, npy_path, k, colormap, metric, atom_name, clamp_min, clamp_max, halfwindow=half_window)

daqcolor_apply_desc = CmdDesc(
    required=[("npy_path", StringArg), ("model", ModelArg)],
    keyword=[("k", IntArg), ("colormap", ColormapArg), ("metric", StringArg),
             ("atom_name", StringArg), ("clamp_min", FloatArg), ("clamp_max", FloatArg), ("half_window", IntArg)],
    synopsis="Color residues once from a numpy (N×32) probability file. " \
    "For metric, you can use " \
    "aa_score          - DAQ(AA) score" \
    "atom_score        - DAQ(CA) score" \
    "aa_conf:[AA type] - DAQ(Selected AA type) score"
)

def daqcolor_monitor(session, model, *, npy_path=None, k=1, colormap=None,
                     metric="aa_score", atom_name="CA", half_window=9, on=True, interval=0.5):
    key = (session, model.id_string)
    if on:
        if npy_path is None:
            raise ValueError("npy_path must be provided when turning monitor on.")

        # Remove existing monitor if one exists to prevent handler leak
        existing = _MON.get(key)
        if existing and "handler" in existing:
            session.triggers.remove_handler(existing["handler"])
            session.logger.info("daqcolor monitor: replacing existing monitor")

        _recolor(session, model, npy_path, k, colormap, metric, atom_name, None, None, halfwindow=half_window)

        import time
        last_update = [time.time()]  # Use list to allow modification in nested function

        def _tick(trigger_name, change_info):
            try:
                # Check if model is still valid (not deleted)
                if model.deleted:
                    # Model was deleted, remove the handler
                    info = _MON.pop(key, None)
                    if info and "handler" in info:
                        session.triggers.remove_handler(info["handler"])
                    session.logger.info("daqcolor monitor stopped (model deleted)")
                    return

                # Throttle updates based on interval
                current_time = time.time()
                if current_time - last_update[0] >= interval:
                    _recolor(session, model, npy_path, k, colormap, metric, atom_name, None, None, halfwindow=half_window)
                    last_update[0] = current_time
            except Exception as e:
                session.logger.warning(f"daqcolor monitor error: {e}")

        h = session.triggers.add_handler("new frame", _tick)
        _MON[key] = {"handler": h}
        session.logger.info(f"daqcolor monitor ON (recolor every {interval}s)")
    else:
        info = _MON.pop(key, None)
        if info and "handler" in info:
            session.triggers.remove_handler(info["handler"])
            session.logger.info("daqcolor monitor OFF")

daqcolor_monitor_desc = CmdDesc(
    required=[("model", ModelArg)],
    keyword=[("npy_path", StringArg), ("k", IntArg), ("colormap", ColormapArg),
             ("metric", StringArg), ("atom_name", StringArg),("half_window", IntArg), ("on", BoolArg), ("interval", FloatArg)],
    synopsis="Start/stop live recoloring with throttling. interval (default 0.5s) controls update frequency. Use 'on false' to stop monitoring."
)

# --- add: show points as markers --------------------------------------------
import os
from chimerax.markers import MarkerSet


def daqcolor_points(session, npy_path, *, radius=0.4, metric=None, colormap=None,
                    clamp_min=None, clamp_max=None):
    import numpy as np
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected (N,>=3) numpy file; got {arr.shape}")

    xyz = arr[:, :3].astype(np.float32)
    N = xyz.shape[0]

    ms = MarkerSet(session, name=f"points:{os.path.basename(npy_path)}")
    session.models.add([ms])

    # ---- 色の用意 ----
    if metric is None:
        rgba = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (N, 1))
    else:
        if arr.shape[1] < 23:
            raise ValueError("Please select correct metric")
        aa = arr[:, 3:23].astype(np.float32)

        if metric == "aa_conf":
            scal = aa.max(axis=1)
        elif metric.startswith("aa_top:"):
            AA20 = ["ALA","VAL","PHE","PRO","MET","ILE","LEU","ASP","GLU","LYS",
                    "ARG","SER","THR","TYR","HIS","CYS","ASN","TRP","GLN","GLY"]
            aa3 = metric.split(":",1)[1].upper()
            if aa3 not in AA20:
                raise ValueError(f"Unknown aa code: {aa3}")
            scal = aa[:, AA20.index(aa3)]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        from chimerax.core.colors import Colormap
        if colormap is None:
            colors_rgbs = np.array([[1.0,0.0,0.0,1.0],
                                    [1.0,1.0,1.0,1.0],
                                    [0.0,0.0,1.0,1.0]], dtype=np.float32)
            cmap = Colormap(name="daq_points", colors=colors_rgbs, data_values=[-1.0,0.0,1.0])
        else:
            cmap = colormap

        x = np.clip(scal if (clamp_min is None or clamp_max is None)
                    else np.clip(scal.astype(np.float32), float(clamp_min), float(clamp_max)),
                    -1.0, 1.0)
        rgba_f = cmap.interpolated_rgba(x)                      # float [0,1]
        rgba = (np.clip(rgba_f, 0, 1) * 255).astype(np.uint8)   # uint8 (N,4)

    # ---- マーカー生成（チャンクで高速化）----
    # create_marker は color=(r,g,b,a) を 0..1 で受け取るので正規化する
    CHUNK = 5000
    for s in range(0, N, CHUNK):
        e = min(N, s+CHUNK)
        for c, col in zip(xyz[s:e], rgba[s:e]):
            ms.create_marker(tuple(map(float, c)),
                             radius=float(radius),
                             rgba=tuple((col).tolist())
                            )

    session.logger.status(f"Loaded {N} points as markers (radius={radius})", color="blue")



daqcolor_points_desc = CmdDesc(
    required=[("npy_path", StringArg)],
    keyword=[("radius", FloatArg), ("metric", StringArg),
             ("colormap", ColormapArg), ("clamp_min", FloatArg), ("clamp_max", FloatArg)],
    synopsis="Show xyz points from a numpy file as markers"
)
# ---------------------------------------------------------------------------

from chimerax.core.commands import CmdDesc

def daqcolor_clear(session):
    # clear points
    to_close = [m for m in session.models if getattr(m, 'name', '').startswith('points:')]
    if to_close:
        session.models.close(to_close)
        session.logger.status(f"Closed {len(to_close)} point model(s).", color="blue")
    else:
        session.logger.status("No point models to close.", color="blue")

daqcolor_clear_desc = CmdDesc(
    required=[],
    synopsis="Close all marker models created by 'daqcolor points'"
)


# ===========================================================================
# DAQ Score Computation Commands
# ===========================================================================

from chimerax.core.commands import OpenFileNameArg, SaveFileNameArg, Or
from chimerax.map import MapArg


def daqscore_compute(session, map_input, contour, *, output=None, stride=2,
                     batch_size=512, max_points=500000, model=None,
                     monitor=None, metric="aa_score", half_window=9):
    """
    Compute DAQ scores from a cryo-EM map.
    
    Parameters
    ----------
    session : ChimeraX session
    map_input : str or Volume
        Path to input MRC/MAP file OR a ChimeraX Volume model (e.g., #1)
    contour : float
        Contour threshold for density map
    output : str, optional
        Path to save output NPY file (auto-generated if not specified)
    stride : int
        Stride for point sampling (default: 2)
    batch_size : int
        Batch size for inference (default: 512)
    max_points : int
        Maximum number of points (default: 500000)
    model : str, optional
        Path to ONNX model (uses bundled model if not specified)
    monitor : Model, optional
        Structure model to auto-monitor after computation
    metric : str
        Coloring metric for monitoring: "aa_score", "atom_score", or "aa_conf:XXX"
    half_window : int
        Half window size for score smoothing (default: 9)
    """
    from pathlib import Path
    from chimerax.map import Volume
    
    # Check for onnxruntime
    try:
        import onnxruntime
    except ImportError:
        session.logger.error(
            "onnxruntime is not installed. Please install it with:\n"
            "  pip install onnxruntime"
        )
        return
    
    from .compute import compute_daq_scores
    
    # Determine if input is a Volume model or file path
    if isinstance(map_input, Volume):
        # Input is an already-loaded ChimeraX Volume
        volume = map_input
        map_name = volume.name or f"volume_{volume.id_string}"
        session.logger.info(f"Computing DAQ scores for volume: #{volume.id_string} ({map_name})")
        
        # Generate output path if not specified
        if output is None:
            # Use current working directory with volume name
            output = Path.cwd() / f"{map_name.replace(' ', '_')}_daq_scores.npy"
        
        map_source = volume  # Pass Volume object directly
    else:
        # Input is a file path
        map_path = Path(map_input)
        session.logger.info(f"Computing DAQ scores for file: {map_path}")
        
        # Generate output path if not specified
        if output is None:
            output = map_path.parent / f"{map_path.stem}_daq_scores.npy"
        
        map_source = map_path  # Pass path
    
    session.logger.info(f"  Output: {output}")
    session.logger.info(f"  Contour: {contour}, Stride: {stride}")
    
    try:
        points, scores, actual_output_path = compute_daq_scores(
            session,
            map_source,
            output_path=output,
            contour=contour,
            stride=stride,
            batch_size=batch_size,
            max_points=max_points,
            model_path=model,
        )

        # Use the actual output path (may differ from requested if fallback was used)
        saved_path = actual_output_path if actual_output_path else output

        session.logger.info(f"DAQ score computation completed!")
        session.logger.info(f"  Points: {points.shape[0]}")
        session.logger.info(f"  Output saved to: {saved_path}")

        # Auto-monitor if structure specified
        if monitor is not None:
            session.logger.info(f"Starting auto-monitor for structure #{monitor.id_string}...")
            # Apply initial coloring
            _recolor(session, monitor, str(saved_path), 1, None, metric, "CA",
                     None, None, halfwindow=half_window)
            # Start monitoring (with default 0.5s interval)
            daqcolor_monitor(session, monitor, npy_path=str(saved_path), k=1, colormap=None,
                           metric=metric, atom_name="CA", half_window=half_window, on=True, interval=0.5)

        return str(saved_path)
        
    except Exception as e:
        session.logger.error(f"DAQ score computation failed: {e}")
        raise


daqscore_compute_desc = CmdDesc(
    required=[("map_input", Or(MapArg, OpenFileNameArg)), ("contour", FloatArg)],
    keyword=[
        ("output", SaveFileNameArg),
        ("stride", IntArg),
        ("batch_size", IntArg),
        ("max_points", IntArg),
        ("model", OpenFileNameArg),
        ("monitor", ModelArg),
        ("metric", StringArg),
        ("half_window", IntArg),
    ],
    synopsis="Compute DAQ scores from a cryo-EM map (file path or loaded volume #id)"
)


def daqscore_run(session, map_input, contour, structure, *, output=None, stride=2,
                 batch_size=512, max_points=500000, model=None,
                 metric="aa_score", k=1, colormap=None, half_window=9):
    """
    Compute DAQ scores and apply coloring to a structure in one step.
    
    Parameters
    ----------
    session : ChimeraX session
    map_input : str or Volume
        Path to input MRC/MAP file OR a ChimeraX Volume model (e.g., #1)
    contour : float
        Contour threshold for density map
    structure : Model
        Structure model to color
    output : str, optional
        Path to save output NPY file
    stride : int
        Stride for point sampling (default: 2)
    batch_size : int
        Batch size for inference (default: 512)
    max_points : int
        Maximum number of points (default: 500000)
    model : str, optional
        Path to ONNX model
    metric : str
        Coloring metric: "aa_score", "atom_score", or "aa_conf:XXX"
    k : int
        Number of nearest neighbors for coloring (default: 1)
    colormap : Colormap, optional
        Color map for visualization
    half_window : int
        Half window size for score smoothing (default: 9)
    """
    from pathlib import Path
    from chimerax.map import Volume
    
    # Check for onnxruntime
    try:
        import onnxruntime
    except ImportError:
        session.logger.error(
            "onnxruntime is not installed. Please install it with:\n"
            "  pip install onnxruntime"
        )
        return
    
    from .compute import compute_daq_scores
    
    # Determine if input is a Volume model or file path
    if isinstance(map_input, Volume):
        volume = map_input
        map_name = volume.name or f"volume_{volume.id_string}"
        session.logger.info(f"Computing DAQ scores for volume: #{volume.id_string}")
        
        if output is None:
            output = Path.cwd() / f"{map_name.replace(' ', '_')}_daq_scores.npy"
        
        map_source = volume
    else:
        map_path = Path(map_input)
        session.logger.info(f"Computing DAQ scores for file: {map_path}")
        
        if output is None:
            output = map_path.parent / f"{map_path.stem}_daq_scores.npy"
        
        map_source = map_path
    
    session.logger.info(f"Computing DAQ scores and applying to structure #{structure.id_string}...")
    
    try:
        # Step 1: Compute DAQ scores
        points, scores, actual_output_path = compute_daq_scores(
            session,
            map_source,
            output_path=output,
            contour=contour,
            stride=stride,
            batch_size=batch_size,
            max_points=max_points,
            model_path=model,
        )

        # Use the actual output path (may differ from requested if fallback was used)
        saved_path = actual_output_path if actual_output_path else output

        # Step 2: Apply coloring to structure
        session.logger.info(f"Applying DAQ coloring to structure #{structure.id_string}...")
        _recolor(session, structure, str(saved_path), k, colormap, metric, "CA",
                 None, None, halfwindow=half_window)

        session.logger.info(f"DAQ score computation and coloring completed!")

        return str(saved_path)

    except Exception as e:
        session.logger.error(f"DAQ score computation failed: {e}")
        raise


daqscore_run_desc = CmdDesc(
    required=[("map_input", Or(MapArg, OpenFileNameArg)), ("contour", FloatArg), ("structure", ModelArg)],
    keyword=[
        ("output", SaveFileNameArg),
        ("stride", IntArg),
        ("batch_size", IntArg),
        ("max_points", IntArg),
        ("model", OpenFileNameArg),
        ("metric", StringArg),
        ("k", IntArg),
        ("colormap", ColormapArg),
        ("half_window", IntArg),
    ],
    synopsis="Compute DAQ scores and apply coloring to a structure (accepts file path or volume #id)"
)

