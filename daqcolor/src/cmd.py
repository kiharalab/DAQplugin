# vim: set expandtab shiftwidth=4 softtabstop=4:
import numpy as np
from chimerax.core.commands import (CmdDesc, StringArg, IntArg, BoolArg,
                                    ColormapArg, FloatArg, ModelArg)

_KDTREE_CACHE = {}  # key -> cKDTree
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


# kNN search
# Add tree
def _knn_idx(db_pts, q_pts, k=8, radius=None, chunk=2000, tree=None):
    try:
        from scipy.spatial import cKDTree
        if tree is None:
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


def _aggregate(pts, aa, q, k=1, radius=None, tree=None):
    """
    pts: (N,3)
    aa:  (N,C) 
    q:   (M,3)
    k:   int

    return:
      aa_nn:        (M,C) 
      has_neighbor: (M,)  
    """
    dist, idx = _knn_idx(pts, q, k=k, radius=radius,tree=tree)  # dist:(M,k), idx:(M,k)
    N, C = aa.shape
    M = q.shape[0]

    if k == 1:
        idx0 = idx[:, 0]
        d0   = dist[:, 0]
        valid = (idx0 >= 0) & (idx0 < N) & np.isfinite(d0)
        aa_nn = np.zeros((M, C), dtype=aa.dtype)
        aa_nn[valid] = aa[idx0[valid]]
        return aa_nn, valid

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
    ss3 = arr[:, 29:32]  # SS : 0,1,2 = helix, sheet, coil

    key = (npy_path, os.path.getmtime(npy_path), pts.shape[0])
    tree = _KDTREE_CACHE.get(key)

    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _KDTREE_CACHE.clear()          # keep only one tree
        _KDTREE_CACHE[key] = tree

    residues = model.residues
    if residues is None or len(residues) == 0:
        session.logger.warning("No residues in model.")
        return

    q = _residue_coords(residues, atom_name=atom_name, use_scene=True)  # (M,3)
    aa_mean, has_nbr = _aggregate(pts, aa, q, k=k, radius=radius, tree=tree)
    atom_mean, has_nbr = _aggregate(pts, atom, q, k=k, radius=radius, tree=tree)
    ss_mean, has_nbr  = _aggregate(pts, ss3,  q, k=k, radius=radius, tree=tree)
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
    elif metric == "ss_score":
        # ss3 の列順を [HELIX, STRAND, COIL] と仮定（必要なら並べ替え）
        scal = np.full((len(residues),), np.nan, dtype=np.float32)

        # 残基ごとに列indexを作る
        idx = np.empty((len(residues),), dtype=np.int32)
        for i, r in enumerate(residues):
            if r.ss_type == r.SS_HELIX:
                idx[i] = 0
            elif r.ss_type == r.SS_STRAND:
                idx[i] = 1
            else:
                idx[i] = 2  # COIL or unknown

        scal[:] = ss_mean[np.arange(len(residues)), idx]
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
                     metric="aa_score", atom_name="CA", 
                     clamp_min=None, clamp_max=None, half_window=9, on=True, interval=0.5):
    key = (session, model.id_string)
    if on:
        if npy_path is None:
            raise ValueError("npy_path must be provided when turning monitor on.")

        # Remove existing monitor if one exists to prevent handler leak
        existing = _MON.get(key)
        if existing and "handler" in existing:
            session.triggers.remove_handler(existing["handler"])
            session.logger.info("daqcolor monitor: replacing existing monitor")

        _recolor(session, model, npy_path, k, colormap, metric, atom_name, clamp_min, clamp_max, halfwindow=half_window)

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
                    _recolor(session, model, npy_path, k, colormap, metric, atom_name, clamp_min, clamp_max, halfwindow=half_window)
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
    keyword=[("npy_path", StringArg), 
             ("k", IntArg), 
             ("colormap", ColormapArg),
             ("metric", StringArg), 
             ("atom_name", StringArg),
             ("clamp_min", FloatArg), ("clamp_max", FloatArg),
             ("half_window", IntArg), 
             ("on", BoolArg), 
             ("interval", FloatArg)
             ],
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


def daqscore_compute_grid(session, map_input, contour, *, structure=None, output=None,
                             stride=2, batch_size=512, max_points=500000, ckpt=None,
                             metric="aa_score", k=1, colormap=None, half_window=9,
                             monitor=False):
    """
    Compute DAQ scores from a cryo-EM map.

    Parameters
    ----------
    session : ChimeraX session
    map_input : str or Volume
        Path to input MRC/MAP file OR a ChimeraX Volume model (e.g., #1)
    contour : float
        Contour threshold for density map
    structure : Model, optional
        Structure model to apply coloring after computation
    output : str, optional
        Path to save output NPY file (auto-generated if not specified)
    stride : int
        Stride for point sampling (default: 2)
    batch_size : int
        Batch size for inference (default: 512)
    max_points : int
        Maximum number of points (default: 500000)
    ckpt : str, optional
        Path to ONNX checkpoint/model file (uses bundled model if not specified)
    metric : str
        Coloring metric: "aa_score", "atom_score", or "aa_conf:XXX"
    k : int
        Number of nearest neighbors for coloring (default: 1)
    colormap : Colormap, optional
        Color map for visualization
    half_window : int
        Half window size for score smoothing (default: 9)
    monitor : bool
        If True and structure is specified, start live monitoring (default: False)
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
            model_path=ckpt,
        )

        # Use the actual output path (may differ from requested if fallback was used)
        saved_path = actual_output_path if actual_output_path else output

        session.logger.info(f"DAQ score computation completed!")
        session.logger.info(f"  Points: {points.shape[0]}")
        session.logger.info(f"  Output saved to: {saved_path}")

        # Apply coloring if structure specified
        if structure is not None:
            session.logger.info(f"Applying DAQ coloring to structure #{structure.id_string}...")
            _recolor(session, structure, str(saved_path), k, colormap, metric, "CA",
                     None, None, halfwindow=half_window)

            # Start monitoring if requested
            if monitor:
                session.logger.info(f"Starting monitor for structure #{structure.id_string}...")
                daqcolor_monitor(session, structure, npy_path=str(saved_path), k=k, colormap=colormap,
                               metric=metric, atom_name="CA", half_window=half_window, on=True, interval=0.5)

        return str(saved_path)
        
    except Exception as e:
        session.logger.error(f"DAQ score computation failed: {e}")
        raise


daqscore_compute_grid_desc = CmdDesc(
    required=[("map_input", Or(MapArg, OpenFileNameArg)), ("contour", FloatArg)],
    keyword=[
        ("structure", ModelArg),
        ("output", SaveFileNameArg),
        ("stride", IntArg),
        ("batch_size", IntArg),
        ("max_points", IntArg),
        ("ckpt", OpenFileNameArg),
        ("metric", StringArg),
        ("k", IntArg),
        ("colormap", ColormapArg),
        ("half_window", IntArg),
        ("monitor", BoolArg),
    ],
    synopsis="Compute DAQ scores from a cryo-EM map (file path or loaded volume #id)"
)




# ===========================================================================
# DAQ Score PDB-based Computation Command
# ===========================================================================

def daqscore_compute_pdb(session, map_input, *, structure=None, output=None,
                         batch_size=512, ckpt=None, metric="aa_score",
                         k=1, colormap=None, half_window=9, apply_color=True,
                         save_model=None):
    """
    Compute DAQ scores using heavy atom positions from a PDB structure.

    This version uses heavy atom coordinates from the structure as query points
    instead of grid points from the map. Reference distributions are computed
    from atoms with density >= 0.

    Parameters
    ----------
    session : ChimeraX session
    map_input : str or Volume
        Path to input MRC/MAP file OR a ChimeraX Volume model (e.g., #1)
    structure : Model
        Structure model whose heavy atom coordinates will be used
    output : str, optional
        Path to save output NPY file (auto-generated if not specified)
    batch_size : int
        Batch size for inference (default: 512)
    ckpt : str, optional
        Path to ONNX checkpoint/model file (uses bundled model if not specified)
    metric : str
        Coloring metric: "aa_score", "atom_score", or "aa_conf:XXX"
    k : int
        Number of nearest neighbors for coloring (default: 1)
    colormap : Colormap, optional
        Color map for visualization
    half_window : int
        Half window size for score smoothing (default: 9)
    apply_color : bool
        If True, apply coloring to structure after computation (default: True)
    save_model : str, optional
        Path to save the scored structure model (PDB or CIF format). 
        Scores are written to B-factor field. If not specified, model is not saved.
    """
    from pathlib import Path
    from chimerax.map import Volume

    # Check structure is provided
    if structure is None:
        session.logger.error("structure argument is required")
        return

    # Check for onnxruntime
    try:
        import onnxruntime
    except ImportError:
        session.logger.error(
            "onnxruntime is not installed. Please install it with:\n"
            "  pip install onnxruntime"
        )
        return

    from .compute import compute_daq_scores_pdb

    # Determine if input is a Volume model or file path
    if isinstance(map_input, Volume):
        volume = map_input
        map_name = volume.name or f"volume_{volume.id_string}"
        session.logger.info(f"Computing DAQ scores (PDB mode) for volume: #{volume.id_string}")

        if output is None:
            output = Path.cwd() / f"{map_name.replace(' ', '_')}_{structure.name}_pdb_daq_scores.npy"

        map_source = volume
    else:
        map_path = Path(map_input)
        session.logger.info(f"Computing DAQ scores (PDB mode) for file: {map_path}")

        if output is None:
            output = map_path.parent / f"{map_path.stem}_{structure.name}_pdb_daq_scores.npy"

        map_source = map_path

    session.logger.info(f"  Structure: #{structure.id_string} ({structure.name})")
    session.logger.info(f"  Output: {output}")

    try:
        # Compute DAQ scores at heavy atom positions
        points, scores, actual_output_path = compute_daq_scores_pdb(
            session,
            map_source,
            structure,
            output_path=output,
            batch_size=batch_size,
            model_path=ckpt,
        )

        # Use the actual output path (may differ from requested if fallback was used)
        saved_path = actual_output_path if actual_output_path else output

        session.logger.info(f"DAQ score computation (PDB mode) completed!")
        session.logger.info(f"  Heavy atoms: {points.shape[0]}")
        session.logger.info(f"  Output saved to: {saved_path}")

        # Apply coloring to structure if requested
        if apply_color:
            session.logger.info(f"Applying DAQ coloring to structure #{structure.id_string}...")
            _recolor(session, structure, str(saved_path), k, colormap, metric, "CA",
                     None, None, halfwindow=half_window)

        # Save model if requested
        if save_model is not None:
            save_path = Path(save_model)
            session.logger.info(f"Saving scored structure to: {save_path}")
            # If apply_color was False, we still need to apply scores to B-factors for saving
            if not apply_color:
                session.logger.info(f"Applying DAQ scores to B-factors for saving...")
                _recolor(session, structure, str(saved_path), k, colormap, metric, "CA",
                         None, None, halfwindow=half_window)
            try:
                # Use ChimeraX save command to write the structure with B-factors
                from chimerax.core.commands import run
                run(session, f"save {save_path} #{structure.id_string}")
                session.logger.info(f"Scored structure saved successfully")
            except Exception as e:
                session.logger.error(f"Failed to save structure: {e}")
                raise

        return str(saved_path)

    except Exception as e:
        session.logger.error(f"DAQ score computation (PDB mode) failed: {e}")
        raise


daqscore_compute_pdb_desc = CmdDesc(
    required=[("map_input", Or(MapArg, OpenFileNameArg))],
    keyword=[
        ("structure", ModelArg),
        ("output", SaveFileNameArg),
        ("batch_size", IntArg),
        ("ckpt", OpenFileNameArg),
        ("metric", StringArg),
        ("k", IntArg),
        ("colormap", ColormapArg),
        ("half_window", IntArg),
        ("apply_color", BoolArg),
        ("save_model", SaveFileNameArg),
    ],
    required_arguments=["structure"],
    synopsis="Compute DAQ scores at heavy atom positions from a PDB structure"
)

# arrowwin functions--------------------------------
import numpy as np
from chimerax.core.commands import CmdDesc, register
from chimerax.core.commands import FloatArg, IntArg, StringArg
from chimerax.core.errors import UserError
from chimerax.core.commands import run

def _ca_xyz(res):
    a = res.find_atom("CA")
    if a is None:
        return None
    c = a.scene_coord
    return np.array([c.x, c.y, c.z], float)



def _get_chain_residues_in_order(structure, chain_id):
    # ChimeraX側の residue order を保って chain ごとに並べる
    # 速度優先でシンプルに：structure.residues を走査して chain_id でフィルタ
    # （必要なら polymer だけに限定）
    return [r for r in structure.residues if r.chain_id == chain_id]

# ----------------------------
# Precompute per-chain nearest grid + scores
# ----------------------------
def precompute_chain_nn_and_scores(residues_all, kdtree, aa_scores):
    """
    residues_all: ordered chain residues (length L)

    Returns:
      ca_xyz  : (L,3) float, CA coords (nan if missing)
      has_ca  : (L,) bool
      nn_idx  : (L,) int, nearest pts index (-1 if missing)
      score20 : (L,20) float, aa_scores[nn_idx, :] (nan if missing)
    """
    L = len(residues_all)
    ca_xyz = np.full((L, 3), np.nan, dtype=float)
    has_ca = np.zeros(L, dtype=bool)

    for j, r in enumerate(residues_all):
        x0 = _residue_coords([r], atom_name="CA", use_scene=True)[0]
        if x0 is None:
            continue
        ca_xyz[j] = x0.astype(float)
        has_ca[j] = True

    nn_idx = np.full(L, -1, dtype=int)
    if np.any(has_ca):
        _, nn = kdtree.query(ca_xyz[has_ca])
        nn_idx[has_ca] = nn.astype(int)

    score20 = np.full((L, aa_scores.shape[1]), np.nan, dtype=float)
    ok = nn_idx >= 0
    score20[ok] = aa_scores[nn_idx[ok], :]

    return ca_xyz, has_ca, nn_idx, score20


import numpy as np
from chimerax.core.commands import run

# ----------------------------
# FAST arrowwin for one residue (NO KNN inside)
# ----------------------------
def add_window_arrow_for_residue(
    session,
    residues_all,
    target_residue,
    ca_xyz, has_ca, score20,      # ★ precomputed
    N_window=2,
    K_shift=3,
    min_move=0.8,
    radius=0.35,
    vmax_color=1.0,
    min_improvement=0.0,
    name_prefix="daq_arrow",
    group=None,
    vmax_radius=1.0,
    max_radius_scale=3.0,
    min_radius_scale=0.5,
):
    """
    - Candidate moves: residue-index shifts sh in {-K..-1, +1..+K}.
    - When evaluating sh, window residues are also shifted by sh in index (NOT translation).
      Residue j keeps its AA type, but is scored at CA(j+sh).
    - Draws a cone from CA(i) to CA(i+best_shift) (full vector).
    """

    AA20 = ["ALA","VAL","PHE","PRO","MET","ILE","LEU","ASP","GLU","LYS",
            "ARG","SER","THR","TYR","HIS","CYS","ASN","TRP","GLN","GLY"]
    AA_INDEX = {aa: i for i, aa in enumerate(AA20)}

    def _arrow_name(res):
        st = getattr(res, "structure", None)
        stid = getattr(st, "id_string", None) or getattr(st, "id", None) or "st"
        ch = getattr(res, "chain_id", None) or "?"
        num = getattr(res, "number", None)
        ins = getattr(res, "insertion_code", "") or ""
        return f"{name_prefix}_{stid}_{ch}_{num}{ins}"

    def score_to_red_white(score, vmax=1.0):
        if score <= 0:
            return "#ffffff"
        x = min(score / vmax, 1.0)  # 0..1
        r = 1.0
        g = 1.0 - x
        b = 1.0 - x
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

    # target index
    try:
        i = residues_all.index(target_residue)
    except ValueError:
        return
    if i < 0 or i >= len(residues_all) or (not has_ca[i]):
        return

    xi = ca_xyz[i]
    col_i = AA_INDEX.get((target_residue.name or "").upper())
    if col_i is None:
        return

    # window indices
    s = max(0, i - int(N_window))
    e = min(len(residues_all), i + int(N_window) + 1)
    win_idx0 = np.arange(s, e, dtype=int)

    # keep those with CA
    win_idx0 = win_idx0[has_ca[win_idx0]]
    if win_idx0.size == 0:
        return

    # map each window residue to AA col + baseline s0 at CA(j)
    win_idx = []
    win_cols = []
    win_s0 = []
    for j in win_idx0.tolist():
        col = AA_INDEX.get((residues_all[j].name or "").upper())
        if col is None:
            continue
        s0 = score20[j, col]
        if not np.isfinite(s0):
            continue
        win_idx.append(j)
        win_cols.append(col)
        win_s0.append(s0)

    if not win_idx:
        return

    win_idx = np.array(win_idx, dtype=int)
    win_cols = np.array(win_cols, dtype=int)
    win_s0 = np.array(win_s0, dtype=float)

    # search best shift
    best_total = -1e18
    best_shift = None

    for sh in range(-int(K_shift), int(K_shift) + 1):
        if sh == 0:
            continue

        tgt_idx = win_idx + sh
        n = len(residues_all)

        in_bounds = (tgt_idx >= 0) & (tgt_idx < n)
        if not np.any(in_bounds):
            continue

        ok = np.zeros_like(in_bounds, dtype=bool)
        ok[in_bounds] = has_ca[tgt_idx[in_bounds]]

        if not np.any(ok):
            continue

        ti = tgt_idx[ok]
        cols = win_cols[ok]
        s0s = win_s0[ok]

        s1 = score20[ti, cols]
        ok1 = np.isfinite(s1)
        if not np.any(ok1):
            continue

        total = float(np.sum(s1[ok1] - s0s[ok1]))
        if total > best_total:
            best_total = total
            best_shift = sh

    if best_shift is None or best_total <= 0:
        return

    jbest = i + best_shift
    if jbest < 0 or jbest >= len(residues_all) or (not has_ca[jbest]):
        return

    xbest = ca_xyz[jbest]
    v = xbest - xi
    vnorm = float(np.linalg.norm(v))
    if vnorm < float(min_move):
        return

    end = xi + v  # show full move vector (no normalization)

    avg_improve = best_total / len(win_idx)
    if avg_improve < min_improvement:
        return
    #print(f"Best shift: {best_shift}, total improve: {best_total:.3f}, avg improve: {avg_improve:.3f} min_improvement: {min_improvement}")
    color_hex = score_to_red_white(avg_improve, vmax=vmax_color)

    # Radius scale
    x = min(avg_improve / float(vmax_radius), 1.0)  # 0..1
    scale = float(min_radius_scale) + x * (float(max_radius_scale) - float(min_radius_scale))
    radius_draw = float(radius) * scale

    name = _arrow_name(target_residue)

    before = set(session.models.list())
    run(session,
        f"shape cone fromPoint {xi[0]},{xi[1]},{xi[2]} "
        f"toPoint {end[0]},{end[1]},{end[2]} "
        f"radius {radius_draw} color {color_hex} name {name}_{avg_improve:.1f} divisions 4",
        log=False
    )

    if group is not None:
        after = set(session.models.list())
        for m in (after - before):
            try:
                session.models.remove([m])
                session.models.add([m], parent=group)
            except Exception:
                pass

from chimerax.core.errors import UserError


def _selected_residues(session):
    from chimerax.atomic import selected_atoms

    atoms = selected_atoms(session)
    if atoms is None or len(atoms) == 0:
        return []
    return list({a.residue for a in atoms if a.residue is not None})

def reset_arrow_group(session, name="DAQ Arrows"):
    # remove old group if exists
    old = [m for m in session.models.list() if m.name == name]
    if old:
        session.models.remove(old)

    # make a new empty group
    group = Model(name, session)
    session.models.add([group])
    return group

def _selected_residues_by_structure(session):
    """
    Returns: dict[AtomicStructure] -> list[Residue]
    """
    from chimerax.atomic import selected_atoms
    atoms = selected_atoms(session)
    if atoms is None or len(atoms) == 0:
        return {}

    by_st = {}
    for a in atoms:
        r = a.residue
        if r is None:
            continue
        st = r.structure
        by_st.setdefault(st, set()).add(r)

    # stable order not guaranteed; keep as list
    return {st: list(res_set) for st, res_set in by_st.items()}

# ---- ChimeraX command ----
# ----------------------------
# Command entry: chain-cache + group reset
# ----------------------------
from chimerax.atomic import AtomicStructure
from chimerax.core.errors import UserError

def daq_arrowwin(
    session,
    structure: AtomicStructure = None,   # optional
    npy_path: str = None,
    chain: str = None,                  # optional chain filter for full-model mode
    nwin: int = 2,
    kshift: int = 3,
    minmove: float = 0.8,
    radius: float = 0.35,
    vmax_color: float = 1.0,
    min_improvement: float = 0.0,
    group_name: str = "DAQ Arrows",
    vmax_radius: float = 1.0,
    max_radius_scale: float = 3.0,
    min_radius_scale: float = 0.5,
    progress_callback=None,        # ★追加（compute.pyと同じ思想）
    update_every: int = 25,        # ★追加（進捗更新の間引き）
    is_cancelled=None,             # ★任意：キャンセルルフラグ（例: threading.Event()）
):
    """
    Behavior:
      - If residues are selected: draw arrows only on those residues (selection mode).
      - Else: draw arrows on all residues in 'structure' (model mode).
      - If selection exists and structure is given, selection still wins.
    """

    def update_progress(cur, tot, msg=""):
        if progress_callback:
            progress_callback(cur, tot, msg)
        else:
            # same as compute.py
            if msg:
                session.logger.status(f"{msg} ({cur}/{tot})")
            else:
                session.logger.status(f"({cur}/{tot})")

    if npy_path is None:
        raise UserError("Please specify npy_path, e.g. 'daq arrowwin #1 scores.npy'.")

    sel_map = _selected_residues_by_structure(session)

    arr = np.load(npy_path)
    pts = arr[:, :3]
    aa  = arr[:, 3:23]

    from scipy.spatial import cKDTree
    kdtree = cKDTree(pts)

    group = reset_arrow_group(session, name=group_name)

    # -----------------------
    # selection mode
    # -----------------------
    if sel_map:
        #compute total for progress
        targets = []
        for st, sel_res in sel_map.items():
            if structure is not None and st is not structure:
                continue
            for r in sel_res:
                ch = r.chain_id
                if chain is not None and ch != chain:
                    continue
                targets.append((st, r))

        total = max(1, len(targets))
        done = 0
        update_progress(0, total, f"Arrow: selection mode ({total} residues")
        # For each structure that has selection, run only those residues.
        for st, sel_res in sel_map.items():
            # optional: if user also specified structure, restrict to that structure only
            if structure is not None and st is not structure:
                continue

            # build per-chain caches inside this structure (only chains needed)
            chain_cache = {}  # chain_id -> precomputed pack

            for r in sel_res:
                ch = r.chain_id
                if chain is not None and ch != chain:
                    continue

                if ch not in chain_cache:
                    residues_all = _get_chain_residues_in_order(st, ch)
                    if not residues_all:
                        continue
                    ca_xyz, has_ca, nn_idx, score20 = precompute_chain_nn_and_scores(residues_all, kdtree, aa)
                    chain_cache[ch] = dict(residues_all=residues_all, ca_xyz=ca_xyz, has_ca=has_ca, score20=score20)

                pack = chain_cache.get(ch)
                if pack is None:
                    continue

                add_window_arrow_for_residue(
                    session,
                    residues_all=pack["residues_all"],
                    target_residue=r,
                    ca_xyz=pack["ca_xyz"],
                    has_ca=pack["has_ca"],
                    score20=pack["score20"],
                    N_window=nwin,
                    K_shift=kshift,
                    min_move=minmove,
                    radius=radius,
                    vmax_color=vmax_color,
                    min_improvement=min_improvement,
                    group=group,
                    vmax_radius=vmax_radius,
                    max_radius_scale=max_radius_scale,
                    min_radius_scale=min_radius_scale,
                    
                )
                done += 1
                if (done % update_every) == 0 or done == total:
                    update_progress(done, total, f"Arrow: drawing ({done}/{total})")

        update_progress(total, total, "Arrow: done")

        return  # done

    # -----------------------
    # model mode (no selection)
    # -----------------------
    if structure is None:
        raise UserError("No residues selected. Specify a structure model, e.g. 'daq arrowwin #1 scores.npy'.")

    chain_ids = sorted({r.chain_id for r in structure.residues})
    if chain is not None:
        chain_ids = [c for c in chain_ids if c == chain]
        if not chain_ids:
            raise UserError(f"Chain '{chain}' not found in {structure}.")

    # check total residues for progress
    chain_res_map = {}
    total = 0
    for ch in chain_ids:
        residues_all = _get_chain_residues_in_order(structure, ch)
        if residues_all:
            chain_res_map[ch] = residues_all
            total += len(residues_all)

    total = max(1, total)
    done = 0
    update_progress(0, total, f"ArrowWin: model mode ({total} residues)")


    for ch in chain_ids:
        residues_all = _get_chain_residues_in_order(structure, ch)
        if not residues_all:
            continue

        ca_xyz, has_ca, nn_idx, score20 = precompute_chain_nn_and_scores(residues_all, kdtree, aa)

        for r in residues_all:
            add_window_arrow_for_residue(
                session,
                residues_all=residues_all,
                target_residue=r,
                ca_xyz=ca_xyz,
                has_ca=has_ca,
                score20=score20,
                N_window=nwin,
                K_shift=kshift,
                min_move=minmove,
                radius=radius,
                vmax_color=vmax_color,
                group=group,
                vmax_radius=vmax_radius,
                max_radius_scale=max_radius_scale,
                min_radius_scale=min_radius_scale,
                min_improvement=min_improvement,
                
            )
            done += 1
            if (done % update_every) == 0 or done == total:
                update_progress(done, total, f"ArrowWin: drawing ({done}/{total})")

    update_progress(total, total, "Arrow: done")
    



from chimerax.core.models import Model

def get_or_create_group(session, name="DAQ Arrows"):
    # reuse existing group if exists
    for m in session.models.list():
        if m.name == name:
            return m

    g = Model(name, session)
    session.models.add([g])          # Add to session (not shown in model list until added)
    return g

def add_cylinder_to_group(session, group, p1, p2, radius=0.4, color=(255,0,0,255)):

    from chimerax.core.commands import run
    x1,y1,z1 = p1; x2,y2,z2 = p2
    run(session, f"shape cylinder {x1},{y1},{z1} {x2},{y2},{z2} radius {radius}")

    cyl = session.models.list()[-1]
    session.models.remove([cyl])                 # 
    session.models.add([cyl], parent=group)      # group

    cyl.color = color
    return cyl




from chimerax.core.commands import CmdDesc, StringArg, IntArg, FloatArg
from chimerax.atomic import AtomicStructureArg

daq_arrowwin_desc = CmdDesc(
    required=[("structure", AtomicStructureArg), ("npy_path", StringArg)],
    keyword=[("chain", StringArg),
             ("nwin", IntArg), ("kshift", IntArg),
             ("minmove", FloatArg), ("radius", FloatArg),
             ("min_improvement", FloatArg),
             ("vmax_color", FloatArg),
             ("vmax_radius", FloatArg),
             ("max_radius_scale", FloatArg),
             ("min_radius_scale", FloatArg),
             ("group_name", StringArg)
             ],
    synopsis="DAQ arrowwin: selection mode if residues selected, otherwise model mode"
)

