# utils.py
import numpy as np


def _residue_coords(residues, atom_name="CA", use_scene=True):
    R = len(residues)

    # Fast path for CA (amino acid principal atom): use vectorized C++ accessor
    # to skip per-residue Python loop + per-atom scene_coord C calls.
    if atom_name == "CA" and R > 0 and hasattr(residues, "existing_principal_atoms"):
        existing = residues.existing_principal_atoms  # Atoms (no Nones)
        if len(existing) == R:
            xyz = existing.scene_coords if use_scene else existing.coords
            return np.asarray(xyz, dtype=np.float32)
        # Some residues have no CA — fall back to per-residue lookup for mapping.
        principal = residues.principal_atoms
        out = np.full((R, 3), np.nan, dtype=np.float32)
        valid_mask = np.fromiter((a is not None for a in principal), dtype=bool, count=R)
        if valid_mask.any():
            xyz = existing.scene_coords if use_scene else existing.coords
            out[valid_mask] = np.asarray(xyz, dtype=np.float32)
        return out

    coords = []
    for r in residues:
        a = r.find_atom(atom_name)
        if a is not None:
            if use_scene:
                coords.append(a.scene_coord)
            else:
                coords.append(a.coord)
        else:
            if atom_name == "CA":
                coords.append((np.nan, np.nan, np.nan))
                continue

            ats = r.atoms
            if len(ats):
                if use_scene:
                    xyz = ats.scene_coords
                else:
                    xyz = ats.coords
                coords.append(xyz.mean(axis=0))
            else:
                coords.append((np.nan, np.nan, np.nan))

    return np.asarray(coords, dtype=np.float32)
