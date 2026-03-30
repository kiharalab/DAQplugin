# arrowwin functions--------------------------------
import numpy as np
from chimerax.core.commands import CmdDesc, register
from chimerax.core.commands import FloatArg, IntArg, StringArg
from chimerax.core.errors import UserError
from chimerax.core.commands import run
from .util import _residue_coords

def _ca_xyz(res):
    a = res.find_atom("CA")
    if a is None:
        return None
    c = a.scene_coord
    return np.array([c.x, c.y, c.z], float)



def _get_chain_residues_in_order(structure, chain_id):

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
        if not np.all(np.isfinite(x0)):
            continue
        ca_xyz[j] = np.asarray(x0, dtype=float)
        has_ca[j] = True

    nn_idx = np.full(L, -1, dtype=int)
    if np.any(has_ca):
        _, nn = kdtree.query(ca_xyz[has_ca])
        nn_idx[has_ca] = nn.astype(int)

    score20 = np.full((L, aa_scores.shape[1]), np.nan, dtype=float)
    ok = nn_idx >= 0
    score20[ok] = aa_scores[nn_idx[ok], :]

    return ca_xyz, has_ca, nn_idx, score20


# ----------------------------
# FAST arrowwin for one residue (NO KNN inside)
# ----------------------------
def _np3(x):
    """ChimeraX座標(tinyarray等)→np.float64(3,)に正規化"""
    a = np.asarray(x, dtype=float)
    if a.shape != (3,):
        a = a.reshape(3,)
    return a

def compute_residue_mapping(
    residues_all,
    target_residue,
    ca_xyz, has_ca, score20,
    N_window=2,
    K_shift=3,
    min_move=0.8,
    min_improvement=0.0,
):
    """
    Returns:
      (res_src, res_dst, avg_improve, xi, xbest) or None
    """
    AA20 = ["ALA","VAL","PHE","PRO","MET","ILE","LEU","ASP","GLU","LYS",
            "ARG","SER","THR","TYR","HIS","CYS","ASN","TRP","GLN","GLY"]
    AA_INDEX = {aa: i for i, aa in enumerate(AA20)}

    try:
        i = residues_all.index(target_residue)
    except ValueError:
        return None
    if i < 0 or i >= len(residues_all) or (not has_ca[i]):
        return None

    xi = ca_xyz[i]
    if AA_INDEX.get((target_residue.name or "").upper()) is None:
        return None

    # window indices
    s = max(0, i - int(N_window))
    e = min(len(residues_all), i + int(N_window) + 1)
    win_idx0 = np.arange(s, e, dtype=int)
    win_idx0 = win_idx0[has_ca[win_idx0]]
    if win_idx0.size == 0:
        return None

    win_idx, win_cols, win_s0 = [], [], []
    for j in win_idx0.tolist():
        col = AA_INDEX.get((residues_all[j].name or "").upper())
        if col is None:
            continue
        s0 = score20[j, col]
        if not np.isfinite(s0):
            continue
        win_idx.append(j); win_cols.append(col); win_s0.append(s0)

    if not win_idx:
        return None

    win_idx = np.array(win_idx, int)
    win_cols = np.array(win_cols, int)
    win_s0   = np.array(win_s0, float)

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
        return None

    jbest = i + best_shift
    if jbest < 0 or jbest >= len(residues_all) or (not has_ca[jbest]):
        return None

    xbest = ca_xyz[jbest]
    v = xbest - xi
    if float(np.linalg.norm(v)) < float(min_move):
        return None

    avg_improve = best_total / len(win_idx)
    if avg_improve < min_improvement:
        return None

    return (target_residue, residues_all[jbest], float(avg_improve), xi, xbest)

def draw_arrow_from_mapping(
    session,
    res_src,
    xi, xbest,
    avg_improve,
    radius=0.35,
    vmax_color=1.0,
    min_improvement=0.0,
    name_prefix="daq_arrow",
    group=None,
    vmax_radius=1.0,
    max_radius_scale=3.0,
    min_radius_scale=0.5,
):
    def _arrow_name(res):
        st = getattr(res, "structure", None)
        stid = getattr(st, "id_string", None) or getattr(st, "id", None) or "st"
        ch = getattr(res, "chain_id", None) or "?"
        num = getattr(res, "number", None)
        ins = getattr(res, "insertion_code", "") or ""
        return f"{name_prefix}_{stid}_{ch}_{num}{ins}"

    def score_to_white_orange(score, vmax=1.0, vmin=0.0):
        if score <= 0:
            return "#ffffff"

        x = min((score - vmin) / (vmax - vmin), 1.0) if vmax > vmin else 0.0

        # orange target = (1.0, 0.55, 0.0)
        r = 1.0
        g = 1.0 - 0.45 * x
        b = 1.0 - 1.0 * x

        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


    color_hex = score_to_white_orange(avg_improve, vmax=vmax_color, vmin=min_improvement)

    x = min(avg_improve / float(vmax_radius), 1.0)
    scale = float(min_radius_scale) + x * (float(max_radius_scale) - float(min_radius_scale))
    radius_draw = float(radius) * scale

    end = xi + (xbest - xi)
    name = _arrow_name(res_src)

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

def _get_isolde_pr_mgr(session, structure):
    """
    ISOLDE が利用可能なら PositionRestraintMgr を返す。
    無ければ None を返す。
    """
    try:
        from chimerax.isolde import session_extensions as sx
    except ImportError:
        session.logger.warning("ISOLDE is not installed. Skipping restraints.")
        return None

    try:
        return sx.get_position_restraint_mgr(structure)
    except Exception:
        session.logger.warning("ISOLDE not initialized for this structure.")
        return None

from chimerax.atomic import Atoms

def _record_daq_restrained_atoms(session, structure, atoms: Atoms):
    d = getattr(session, "_daq_restrained_atoms_by_structure", None)
    if d is None:
        d = {}
        session._daq_restrained_atoms_by_structure = d

    if structure not in d or d[structure] is None:
        d[structure] = atoms
    else:
        d[structure] = d[structure].merge(atoms)


def apply_isolde_restraints_from_mapping(
    pr_mgr,
    session,
    res_src,
    res_dst,
    atom_names=("N","CA","C","CB"),
    spring_constant=1500.0,
    use_scene=True,
    require_same_name=True,
    fallback_to_ca_shift=True,
):
    """
    compare res_src and res_dst
    find the same atom res_src -> res_dst (by name, e.g. CA)
    """
    from chimerax.atomic import Atoms

    if pr_mgr is None:
        return 0

    src_atoms = []
    targets = []

    for an in atom_names:
        a_src = res_src.find_atom(an)
        if a_src is None:
            continue

        a_dst = res_dst.find_atom(an)
        if a_dst is None:
            if require_same_name:
                continue
            else:
                # add fallback?: shift CA by the same vector as res_src->res_dst
                continue

        t = _np3(a_dst.scene_coord if use_scene else a_dst.coord)

        src_atoms.append(a_src)
        targets.append(t)
        #show log
        session.logger.info(f"Applying ISOLDE restraint: {res_src} {an} -> {res_dst} {an}, target {t}, spring {spring_constant}")

    if not src_atoms:
        return 0

    # restraint 作成
    src_atoms = Atoms(src_atoms)
    pr_mgr.add_restraints(src_atoms)

    _record_daq_restrained_atoms(session, res_src.structure, src_atoms)  # session に記録しておく（後で clear するため）

    # target / spring を設定
    created = 0
    for a_src, t in zip(src_atoms, targets):
        pr = pr_mgr.get_restraint(a_src)
        if pr is None:
            continue
        pr.target = (float(t[0]), float(t[1]), float(t[2]))
        pr.spring_constant = float(spring_constant)
        pr.enabled = True
        created += 1

    return created


def daq_clearrestraints(session, structure):
    """
    Clear DAQ-created ISOLDE position restraints (tracked in session._daq_restrained_atoms).
    """

    pr_mgr = _get_isolde_pr_mgr(session, structure)  # 
    if pr_mgr is None:
        return 0
    deleted = clear_daq_position_restraints(session, structure, pr_mgr)
    session.logger.info(f"Deleted {deleted} DAQ position restraints.")
    return deleted

def clear_daq_position_restraints(session, structure, pr_mgr):
    """
    指定 structure に対して、DAQ が記録した atom の position restraints を無効化する。
    """
    d = getattr(session, "_daq_restrained_atoms_by_structure", None)
    if not d or structure not in d or d[structure] is None or len(d[structure]) == 0:
        return 0

    atoms = d[structure]
    disabled = 0

    for a in atoms:
        pr = pr_mgr.get_restraint(a)
        if pr is None:
            continue
        pr.enabled = False
        disabled += 1

    d[structure] = None
    return disabled




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

    def score_to_white_green(score, vmax=1.0,vmin=0.0):
        """
        White → Bright Green (optimized for black background)
        0      → #ffffff
        vmax   → #00ff00
        """
        if score <= 0:
            return "#ffffff"

        x = min((score - vmin) / (vmax - vmin), 1.0) if vmax > vmin else 0.0  # 0..1

        # white = (1,1,1)
        # green = (0,1,0)

        r = 1.0 - x
        g = 1.0
        b = 1.0 - x

        return "#{:02x}{:02x}{:02x}".format(
            int(r * 255),
            int(g * 255),
            int(b * 255),
        )

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
    color_hex = score_to_white_green(avg_improve, vmax=vmax_color,vmin=min_improvement)

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
    spring_constant=1500.0,         # ★追加：ISOLDEのスプリング定数（apply_isolde_restraints_from_mappingで使用
    apply_isolde_restraints=False,     # ★追加：ISOLDEの位置拘束も同時に作成するか
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

    pr_mgr = None
    if apply_isolde_restraints:
        session.logger.info("ISOLDE restraints will be applied. Make sure ISOLDE is installed and initialized for the structure.")
        pr_mgr = _get_isolde_pr_mgr(session, structure)

    if apply_isolde_restraints and pr_mgr is not None:
        n_del = clear_daq_position_restraints(session, structure, pr_mgr)
        if n_del:
            session.logger.info(f"Cleared previous DAQ restraints: {n_del}")

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

                '''
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
                '''
                mapping = compute_residue_mapping(
                    residues_all=pack["residues_all"],
                    target_residue=r,
                    ca_xyz=pack["ca_xyz"],
                    has_ca=pack["has_ca"],
                    score20=pack["score20"],
                    N_window=nwin,
                    K_shift=kshift,
                    min_move=minmove,
                    min_improvement=min_improvement,
                )
                done += 1
                if (done % update_every) == 0 or done == total:
                    update_progress(done, total, f"Arrow(Selected): drawing ({done}/{total})")

                if mapping is None:
                    continue
                res_src, res_dst, avg_improve, xi, xbest = mapping

                # 2) draw
                draw_arrow_from_mapping(
                    session,
                    res_src=res_src,
                    xi=xi, xbest=xbest,
                    avg_improve=avg_improve,
                    radius=radius,
                    vmax_color=vmax_color,
                    min_improvement=min_improvement,
                    group=group,
                    vmax_radius=vmax_radius,
                    max_radius_scale=max_radius_scale,
                    min_radius_scale=min_radius_scale,
                )
                if pr_mgr is not None and apply_isolde_restraints:
                    apply_isolde_restraints_from_mapping(
                        pr_mgr,
                        session,
                        res_src=res_src,
                        res_dst=res_dst,
                        atom_names=("N","CA","C","CB"),
                        spring_constant=spring_constant,
                        use_scene=True,
                        fallback_to_ca_shift=True,
                    )
                

        update_progress(total, total, "Arrow(Selected): done")

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
            '''
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
            '''
            mapping = compute_residue_mapping(
                    residues_all=residues_all,
                    target_residue=r,
                    ca_xyz=ca_xyz,
                    has_ca=has_ca,
                    score20=score20,
                    N_window=nwin,
                    K_shift=kshift,
                    min_move=minmove,
                    min_improvement=min_improvement,
                )
            done += 1
            if (done % update_every) == 0 or done == total:
                update_progress(done, total, f"Arrow(all): drawing ({done}/{total})")
            if mapping is None:
                continue
            res_src, res_dst, avg_improve, xi, xbest = mapping

            # 2) draw
            draw_arrow_from_mapping(
                    session,
                    res_src=res_src,
                    xi=xi, xbest=xbest,
                    avg_improve=avg_improve,
                    radius=radius,
                    vmax_color=vmax_color,
                    min_improvement=min_improvement,
                    group=group,
                    vmax_radius=vmax_radius,
                    max_radius_scale=max_radius_scale,
                    min_radius_scale=min_radius_scale,
            )
            if pr_mgr is not None and apply_isolde_restraints:
                apply_isolde_restraints_from_mapping(
                    pr_mgr,session,
                    res_src=res_src,
                    res_dst=res_dst,
                    atom_names=("N","CA","C","CB"),
                    spring_constant=spring_constant,
                    use_scene=True,
                    fallback_to_ca_shift=True,
                )

    update_progress(total, total, "Arrow(all): done")
    



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

