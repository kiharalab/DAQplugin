# vim: set expandtab shiftwidth=4 softtabstop=4:
import numpy as np
from chimerax.core.commands import (CmdDesc, StringArg, IntArg, BoolArg,
                                    ColormapArg, FloatArg, ModelArg,run)

from .util import _residue_coords

_KDTREE_CACHE = {}  # key -> cKDTree
_NPY_CACHE = {}   # keep only one entry, like _KDTREE_CACHE

# Session Monitor
_MON = {}  # (session, model.id_string) -> dict

# For recolor: cache loaded numpy data and KDTree to avoid reloading/rebuilding on every frame
_RECOLOR_CACHE = {}  # keep only one


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



def _recolor(session, model, npy_path, k, cmap, metric, atom_name, clamp_min, clamp_max, radius=3.0, halfwindow=9, 
             eps=0.1, monitor=False):
    AA20 = [
    "ALA","VAL","PHE","PRO","MET","ILE","LEU","ASP","GLU","LYS",
    "ARG","SER","THR","TYR","HIS","CYS","ASN","TRP","GLN","GLY"
    ]

    AA_INDEX = {aa:i for i,aa in enumerate(AA20)}
    ATOM_TYPES6 = ["Other","N","CA","C","O","CB"]  #（index 0..5）

    mtime = os.path.getmtime(npy_path)
    npy_key = (npy_path, mtime)
    cached = _NPY_CACHE.get(npy_key)
    if cached is None:
        #load numpy file
        print(f"Loading numpy file: {npy_path}")
        arr = np.load(npy_path)
        if arr.ndim != 2 or arr.shape[1] != 32:
            raise ValueError(f"Expected (N,32) numpy file; got {arr.shape}")
        #pts  = arr[:, :3].astype(np.float32)
        #aa   = arr[:, 3:23].astype(np.float32)
        #atom = arr[:, 23:29].astype(np.float32)
        #ss3 = arr[:, 29:32].astype(np.float32)  # SS : 0,1,2 = helix, sheet, coil
        # contiguous + float32
        pts  = np.ascontiguousarray(arr[:, :3],   dtype=np.float32)
        aa   = np.ascontiguousarray(arr[:, 3:23], dtype=np.float32)
        atom = np.ascontiguousarray(arr[:, 23:29],dtype=np.float32)
        ss3  = np.ascontiguousarray(arr[:, 29:32],dtype=np.float32)

        from scipy.spatial import cKDTree
        tree = cKDTree(pts)

        _NPY_CACHE.clear()  # keep only one entry
        _NPY_CACHE[npy_key] = {
            "pts": pts,
            "aa": aa,
            "atom": atom,
            "ss3": ss3,
            "tree": tree
        }
    else:
        pts, aa, atom, ss3 = cached["pts"], cached["aa"], cached["atom"], cached["ss3"]
        tree = cached["tree"]

    residues = model.residues
    if residues is None or len(residues) == 0:
        session.logger.warning("No residues in model.")
        return

    q = _residue_coords(residues, atom_name=atom_name, use_scene=True)  # (M,3)

    # ---- cache key: same npy and paramaters ----
    
    param_key = (npy_path, mtime, k, float(radius), metric, atom_name, halfwindow,
                None if clamp_min is None else float(clamp_min),
                None if clamp_max is None else float(clamp_max),
                model.id_string)

    model_cache = _RECOLOR_CACHE.get(param_key)

    if model_cache is not None:
        prev_q = model_cache["q"]
        # shape check: if the number of residues or atom_name changes, we cannot reuse the cache
        if prev_q.shape == q.shape:
            # max movement
            d2 = np.max(np.sum((q - prev_q)**2, axis=1))
            if d2 <= float(eps)*float(eps):
                if monitor:
                    ri = model_cache["res_idx"]
                    changed = False

                    # ribbon colors check (sampled)
                    if ri is not None and len(ri) > 0:
                        cur_res = residues.ribbon_colors
                        ref_res = model_cache["res_rgba"]
                        if cur_res is None or cur_res.shape != ref_res.shape:
                            changed = True
                        elif not np.array_equal(cur_res[ri], ref_res[ri]):
                            changed = True
                    if changed: #color changed by someone
                        residues.ribbon_colors = model_cache["res_rgba"]
                        residues.atoms.colors  = model_cache["atom_rgba"]
                        residues.atoms.bfactors = model_cache["bf_vals"]

                    return  # skip update if monitor and no significant movement
                
                residues.ribbon_colors = model_cache["res_rgba"]
                residues.atoms.colors  = model_cache["atom_rgba"]
                residues.atoms.bfactors = model_cache["bf_vals"]
                return

    # metric
    if metric == "aa_score":
        aa_mean, has_nbr = _aggregate(pts, aa, q, k=k, radius=radius, tree=tree)
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
        aa_mean, has_nbr = _aggregate(pts, aa, q, k=k, radius=radius, tree=tree)
        aa3 = metric.split(":",1)[1].upper()
        j = AA20.index(aa3)
        scal = aa_mean[:, j]
    elif metric == "atom_score":
        atom_mean, has_nbr = _aggregate(pts, atom, q, k=k, radius=radius, tree=tree)
        j = ATOM_TYPES6.index(atom_name) #index: 2 : CA
        scal = atom_mean[:, j]
    elif metric == "ss_score":
        #Ensure DSSP has been run to assign secondary structure types to residues
        try:
            run(session, f"dssp #{model.id_string}")
        except Exception as e:
            session.logger.warning(f"DSSP failed: {e}. Secondary structure types may be unavailable.")

        ss_mean, has_nbr  = _aggregate(pts, ss3,  q, k=k, radius=radius, tree=tree)
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

    #color cache:
    max_res_samp = 128
    R = len(residues)

    res_idx = np.linspace(0, R-1, min(R, max_res_samp)).astype(np.int32) if R else np.array([], np.int32)

    # bf_vals, res_rgba, atom_rgba
    _RECOLOR_CACHE.clear()  # keep only one
    _RECOLOR_CACHE[param_key] = {
        "q": q.copy(),  # 次回比較用
        "bf_vals": bf_vals.copy(),
        "res_rgba": res_rgba.copy(),
        "atom_rgba": atom_rgba.copy(),
        "res_idx": res_idx.copy()
    }

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
                    _recolor(session, model, npy_path, k, colormap, metric, atom_name, clamp_min, clamp_max, halfwindow=half_window, monitor=True)
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




from chimerax.core.commands import CmdDesc, StringArg, IntArg, FloatArg
from chimerax.atomic import AtomicStructureArg
from .arrow import daq_arrowwin, daq_clearrestraints

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
             ("group_name", StringArg),
             ("apply_isolde_restraints", BoolArg),
             ("spring_constant", FloatArg),
             ],
    synopsis="DAQ arrowwin: selection mode if residues selected, otherwise model mode"
)

daq_clearrestraints_desc = CmdDesc(
    required=[("structure", AtomicStructureArg)],
    synopsis="Clear ISOLDE position restraints created by DAQ arrowwin"
)
