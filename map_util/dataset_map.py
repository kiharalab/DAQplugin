
# =============================================================
# File: dataset_map.py
# Description: Map -> candidate points -> patch extraction (online),
#              ready for DataLoader multiprocessing
# =============================================================
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import math
import json
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset

try:
    import zarr  # pip install zarr
except ImportError:
    zarr = None

try:
    from scipy.spatial import cKDTree  # pip install scipy
except Exception:
    cKDTree = None

from typing import Dict, List, Tuple, Optional
import os
import math



try:
    import zarr  # pip install zarr
except ImportError:
    zarr = None

try:
    from scipy.spatial import cKDTree  # pip install scipy
except Exception:
    cKDTree = None


class MapPointPatchDataset(Dataset):


    def __init__(
        self,
        root: str,
        map_ids: List[str],
        Np: int = 4096,
        patch_vox: int = 23,
        intensity_jitter: float = 0.1,  # +/- fraction
        noise_std: float = 0.05,
        dtype: str = 'float32',
        points_override: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        assert zarr is not None, "zarr is required (pip install zarr)"
        self.root = root
        self.map_ids = map_ids
        self.Np = Np
        self.patch_vox = patch_vox
        self.intensity_jitter = intensity_jitter
        self.noise_std = noise_std
        self.dtype = np.float32 if dtype == 'float32' else np.float16

        # optional external points
        self.points_override = points_override or {}

        # index structures per map
        self._meta: Dict[str, Dict] = {}
        self._zarr: Dict[str, zarr.Group] = {}
        self._points: Dict[str, np.ndarray] = {} # (N,3) in Å
        self._density: Dict[str, np.ndarray] = {}  #density values at points
        

        for mid in self.map_ids:
            g = zarr.open(os.path.join(root, mid, 'map.zarr'), mode='r')
            self._zarr[mid] = g
            with open(os.path.join(root, mid, 'meta.json'), 'r') as f:
                self._meta[mid] = json.load(f)

            if mid in self.points_override:
                # 1) external override points (highest priority)
                pts = self.points_override[mid]
                if not isinstance(pts, np.ndarray):
                    raise TypeError(f"points_override[{mid}] must be a numpy array")
                if not (pts.ndim == 2 and pts.shape[1] == 3):
                    raise ValueError(f"points_override[{mid}] must have shape (N,3)")
                if pts.shape[0] == 0:
                    raise ValueError(f"points_override[{mid}] is empty")

                self._points[mid] = pts.astype(np.float32)
            else:
                pts = np.load(os.path.join(root, mid, 'points.npy'))  # (N,3) in Å
                assert pts.ndim == 2 and pts.shape[1] == 3
                self._points[mid] = pts.astype(np.float32)

                dens_path = os.path.join(root, mid, 'density.npy')     # (N,) density values at points
                dens = np.load(dens_path)                              # (N,)
                assert dens.ndim == 1 and dens.shape[0] == pts.shape[0], \
                    f"density shape mismatch: {dens.shape} vs points {pts.shape}"
                self._density[mid] = dens.astype(np.float32)

        # for sampling across maps with replacement
        self._cum_sizes = np.cumsum([self._points[m].shape[0] for m in self.map_ids])
        self._total_points = int(self._cum_sizes[-1])

    def __len__(self):
        # define an epoch in terms of total samples you want; here use total_points // Np
        return max(1, self._total_points // self.Np)

    # ------------------------------ utils ------------------------------
    @staticmethod
    def _world_to_voxel_xyz(coords_xyz: np.ndarray, origin_xyz: np.ndarray, voxel_size_xyz: np.ndarray) -> np.ndarray:
        """

        """
        return (coords_xyz - origin_xyz) / voxel_size_xyz

    @staticmethod
    def _extract_patch(volume_zyx: np.ndarray, center_zyx: np.ndarray, size: int) -> np.ndarray:
        """
        volume_zyx: (Z,Y,X)
        center_zyx: (z,y,x) のfloat coords
        """
        cz, cy, cx = np.round(center_zyx).astype(int)
        r = size // 2

        z0, z1 = cz - r, cz + r + 1
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1

        patch = np.zeros((size, size, size), dtype=volume_zyx.dtype)

        vz0, vz1 = max(0, z0), min(volume_zyx.shape[0], z1)
        vy0, vy1 = max(0, y0), min(volume_zyx.shape[1], y1)
        vx0, vx1 = max(0, x0), min(volume_zyx.shape[2], x1)

        pz0, pz1 = vz0 - z0, vz1 - z0
        py0, py1 = vy0 - y0, vy1 - y0
        px0, px1 = vx0 - x0, vx1 - x0

        patch[pz0:pz1, py0:py1, px0:px1] = volume_zyx[vz0:vz1, vy0:vy1, vx0:vx1]
        return patch

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        # intensity jitter
        if self.intensity_jitter > 0:
            scale = 1.0 + (np.random.rand() * 2 - 1) * self.intensity_jitter
            patch = patch * scale
        # gaussian-like noise
        if self.noise_std > 0:
            patch = patch + np.random.normal(0.0, self.noise_std, size=patch.shape).astype(patch.dtype)
        return patch

    # ------------------------------ sampling ------------------------------
    def _sample_map_and_points(self) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        # choose map proportional to #points
        r = np.random.randint(0, self._total_points)
        midx = int(np.searchsorted(self._cum_sizes, r, side='right'))
        map_id = self.map_ids[midx]

        pts = self._points[map_id]  # (N,3)

        # sample points within map
        if pts.shape[0] <= self.Np:
            sel = np.arange(pts.shape[0], dtype=np.int64)
        else:
            sel = np.random.choice(pts.shape[0], size=self.Np, replace=False).astype(np.int64)

        pts_sel = pts[sel]  # (Np,3)

        # density: if points_override is used for this map_id, return zeros
        if hasattr(self, "points_override") and (map_id in self.points_override):
            dens_sel = np.zeros((pts_sel.shape[0],), dtype=np.float32)
        else:
            dens = self._density[map_id]  # (N,)
            dens_sel = dens[sel].astype(np.float32)

        return map_id, pts_sel, dens_sel, sel


    # ------------------------------ __getitem__ ------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        map_id, pts_world, dens_raw, sel = self._sample_map_and_points()  # (Np,3) in Å, and density values

        g = self._zarr[map_id]
        volume_zyx = g['volume'][:]  # (Z,Y,X)
        meta = self._meta[map_id]
        origin_xyz = np.asarray(meta['origin'], dtype=np.float32)        # (X,Y,Z) in Å
        voxel_xyz  = np.asarray(meta['voxel_size'], dtype=np.float32)    # (X,Y,Z) in Å/voxel

        # (X,Y,Z) → (X,Y,Z) のfloat voxel座標
        pts_vox_xyz = self._world_to_voxel_xyz(pts_world, origin_xyz, voxel_xyz)  # (Np,3)
        # volumeは(Z,Y,X)なので順序を入れ替える
        pts_vox_zyx = pts_vox_xyz[:, [2, 1, 0]]

        patches = []
        for i in range(pts_world.shape[0]):
            p = self._extract_patch(volume_zyx, pts_vox_zyx[i], self.patch_vox)
            p = self._augment_patch(p)
            patches.append(p)
        patches = np.stack(patches, axis=0)  # [Np, D, H, W]

        patches_t = torch.from_numpy(patches.astype(np.float32)).unsqueeze(1)
        points_t  = torch.from_numpy(pts_world.astype(np.float32))
        density_t = torch.from_numpy(dens_raw.astype(np.float32))
        voxsz_t   = torch.from_numpy(voxel_xyz)   # (X,Y,Z)
        origin_t  = torch.from_numpy(origin_xyz)  # (X,Y,Z)


        return {
            'patches': patches_t,
            'points': points_t,
            'density_raw': density_t,
            'map_id': map_id,
            'voxel_size': voxsz_t,
            'origin': origin_t,
        }




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True, help='CSV/TSV with columns: map_path,cif_path,contour[,out_id]')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--max_points', type=int, default=500000, help='Cap number of points per map (random downsample)')
    ap.add_argument('--stride', type=int, default=1, help='Coarse stride before thresholding to reduce candidates')
    args = ap.parse_args()

    entries = parse_table(args.table)
    os.makedirs(args.out_root, exist_ok=True)

    summary = []
    for e in entries:
        out_dir, npts = build_one(e, out_root=args.out_root, max_points=args.max_points, stride=args.stride)
        summary.append({'out_dir': out_dir, 'n_points': int(npts)})
        print(f"Built {out_dir}  points={npts}")

    with open(os.path.join(args.out_root, 'prep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
