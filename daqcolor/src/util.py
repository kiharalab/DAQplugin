# utils.py
import numpy as np

def _residue_coords(residues, atom_name="CA", use_scene=True):
    coords = []
    for r in residues:
        a = r.find_atom(atom_name)
        if a is not None:
            if use_scene:
                coords.append(a.scene_coord)
            else:
                coords.append(a.coord)
        else:
            # No CA Atom?
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
