
# =============================================================
# File: prep_points_from_mrc.py
# Description: Build zarr map + candidate points (≥ contour) in Å world coords
# Input: a CSV/TSV with columns: map_path,cif_path,contour,out_id
# Output per out_id directory:
#   map.zarr/ (volume)
#   meta.json (voxel_size, origin, stats, contour, source paths)
#   points.npy (N,3) candidate points in Å (world coords)
# Notes:
#  - Handles non-1 Å voxel sizes from MRC header
#  - Origin resolution order: header.origin -> -(nxstart,ystart,zstart)*voxel_size -> [0,0,0]
#  - Supports downsampling to cap #points
# =============================================================
from __future__ import annotations
import os, json, csv, argparse
from typing import Optional
import numpy as np

import mrcfile   # pip install mrcfile
import zarr      # pip install zarr


def read_mrc_with_meta(path: str):
    with mrcfile.open(path, permissive=True) as mrc:
        vol = mrc.data.astype(np.float32)  # shape (Z,Y,X)
        # voxel size in Å/voxel
        try:
            vs = np.array([mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x], dtype=np.float32)
        except Exception:
            # fallback from cell dimensions
            nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
            cella = mrc.header.cella  # in Å
            vs = np.array([cella.z/nz, cella.y/ny, cella.x/nx], dtype=np.float32)
        # origin in Å
        origin = None
        # MRC2014 origin field
        if hasattr(mrc.header, 'origin'):
            org = np.array([mrc.header.origin.z, mrc.header.origin.y, mrc.header.origin.x], dtype=np.float32)
            if np.isfinite(org).all():
                origin = org
        # fallback from start indices
        if origin is None:
            try:
                nxst, nyst, nzst = int(mrc.header.nxstart), int(mrc.header.nystart), int(mrc.header.nzstart)
                origin = -np.array([nzst, nyst, nxst], dtype=np.float32) * vs
            except Exception:
                origin = np.zeros(3, dtype=np.float32)
    return vol, vs, origin

import zarr
from numcodecs import Blosc

#def save_zarr_volume(vol: np.ndarray, out_dir: str, chunks=(64,64,64)):
#    g = zarr.open(os.path.join(out_dir, 'map.zarr'), mode='w')
#    g.create_dataset('volume', data=vol, chunks=chunks, compressor=Blosc(cname='lz4', clevel=1, shuffle=2))

import os
import numpy as np
import zarr
from numcodecs import Blosc

def save_zarr_volume(vol: np.ndarray, out_dir: str, chunks=(64, 64, 64)):
    os.makedirs(out_dir, exist_ok=True)
    store_path = os.path.join(out_dir, "map.zarr")

    g = zarr.open(store_path, mode="w")

    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)

    # zarr v3 互換: shape/dtype を明示して作成 → 代入
    arr = g.create_dataset(
        "volume",
        shape=vol.shape,
        dtype=vol.dtype,
        chunks=chunks,
        compressor=compressor,
        overwrite=True,
    )
    arr[:] = vol  # 書き込み
    print("Saved zarr volume to:", store_path)

    return store_path

import numpy as np
from typing import Optional, Tuple

def threshold_points(
    vol: np.ndarray,
    contour: float,
    voxel_size: np.ndarray,   # (sx, sy, sz) in Å
    origin: np.ndarray,       # (ox, oy, oz) in Å, for index (0,0,0)
    max_points: Optional[int] = None,
    stride: int = 1,
    origin_is_center: bool = True,  # False の場合は +0.5 voxel offset?
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N,3) world coords (Å) for voxels with density ≥ contour.
       vol is indexed as (Z, Y, X); output is (X, Y, Z).
    """
    if stride > 1:
        vol_s = vol[::stride, ::stride, ::stride]
        mask = vol_s >= contour
        idx_zyx = np.argwhere(mask)              # (k,3): z,y,x in strided grid
        if idx_zyx.size == 0:
            return (np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.float32))
        # 元グリッドに戻す
        idx_zyx = idx_zyx * stride
    else:
        mask = vol >= contour
        idx_zyx = np.argwhere(mask)
        if idx_zyx.size == 0:
            return (np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.float32))

    # --- density values at selected voxels ---
    dens = vol[idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]].astype(np.float32)

    # Downsample
    if max_points is not None and idx_zyx.shape[0] > max_points:
        sel = np.random.choice(idx_zyx.shape[0], size=max_points, replace=False)
        idx_zyx = idx_zyx[sel]
        dens = dens[sel]

    # ZYX -> XYZ に並べ替え
    idx_xyz = idx_zyx[:, ::-1].astype(np.float32)   # (x, y, z)

    # 原点がボクセル中心でなく「角」を指す場合は +0.5 の補正を入れる
    center_offset = 0.0 if origin_is_center else 0.5
    idx_xyz += center_offset

    # world = origin + idx_xyz * voxel_size   （各軸別スケール）
    pts = origin[None, :].astype(np.float32) + idx_xyz * voxel_size[None, :].astype(np.float32)
    return pts.astype(np.float32), dens.astype(np.float32)

import numpy as np

def find_top_x(vol: np.ndarray, c: float, nbins: int = 200) -> float:
    """
    DAQ の C 関数 FindTopX と同様のロジックで CutOff density を求める関数。
    
    Parameters
    ----------
    vol : np.ndarray
        3D volume (density values)
    c : float
        0〜1 程度の係数。SumCut = Sum * c を満たす位置を探す。
    nbins : int
        0〜dmax を分割するビン数（C 実装では 200）
    """
    vol = np.asarray(vol, dtype=np.float64)
    
    # dens > 0 のボクセルだけを対象にする（C 実装と同じ）
    dens = vol[vol > 0.0]
    if dens.size == 0:
        return 0.0

    # C 側の m->dmax 相当（ヘッダ値を使うなら別途渡してもよいです）
    dmax = float(dens.max())
    if dmax <= 0.0:
        return 0.0

    tic = dmax / nbins  # 1 ビンの幅
    if tic <= 0.0:
        return 0.0

    # 0〜dmax の範囲で nbins ビンのヒストグラムを計算
    counts, edges = np.histogram(dens, bins=nbins, range=(0.0, dmax))

    # Count[i] > 0 のところだけ log を取り、Sum を計算
    log_counts = np.zeros_like(counts, dtype=np.float64)
    mask_pos = counts > 0
    log_counts[mask_pos] = np.log(counts[mask_pos])

    Sum = float(log_counts[mask_pos].sum())
    if Sum <= 0.0:
        # ログの総和が 0 の場合は、単純に 0 を返しておく
        return 0.0

    SumCut = Sum * c

    # 累積して、SumCut を超える最初のビン i を探す
    cum = 0.0
    cutoff_bin = 0
    for i, lc in enumerate(log_counts):
        if lc > 0.0:
            cum += lc
            if cum >= SumCut:
                cutoff_bin = i
                break

    # C 実装と同じく tic * i を CutOff とする
    cutoff = tic * cutoff_bin
    return float(cutoff)


def build_one(entry, out_root: str, max_points: Optional[int], stride: int):
    map_path = entry['map_path']
    cif_path = entry.get('cif_path','')
    contour = float(entry['contour'])
    out_id = entry.get('out_id') or os.path.splitext(os.path.basename(map_path))[0]

    out_dir = os.path.join(out_root, out_id)
    os.makedirs(out_dir, exist_ok=True)

    vol, voxel_size, origin = read_mrc_with_meta(map_path)
    vol = np.maximum(vol, 0) #value <=0 -> ZERO

    c_top = entry.get('top_c', 0.95)  # entry で指定されていれば使う
    p_low = 0.0
    p_high = find_top_x(vol, c=c_top, nbins=200)

    # 安全対策: p_high が小さすぎる、または 0 の場合はフォールバック
    if p_high <= p_low + 1e-8:
        # フォールバックとして単純な percentile を使うなど
        p_low, p_high = np.percentile(vol[vol > 0], [1.0, 99.0])

    # --- ModelAngelo-style normalization ---
    # 1) 外れ値に頑健なパーセンタイルでクリップ（例: 1–99%）
    #p_low, p_high = np.percentile(vol, [1.0, 99.0])
    vol_clip = np.clip(vol, p_low, p_high)

    # 2) min–max で [0,1] にスケール
    vmin = float(vol_clip.min())
    vmax = float(vol_clip.max())
    vol_norm = (vol_clip - vmin) / (vmax - vmin + 1e-8)

    # 3) 保存（学習は map.zarr/volume の [0,1] を入力にする）
    save_zarr_volume(vol_norm.astype(np.float32), out_dir)

    # points は従来通り RAW から抽出（等値面の一貫性を保つ）
    pts, dens = threshold_points(vol, contour=contour, voxel_size=voxel_size, origin=origin,
                           max_points=max_points, stride=stride)
    np.save(os.path.join(out_dir, 'points.npy'), pts)
    np.save(os.path.join(out_dir, 'density.npy'), dens)

    meta = {
        'voxel_size': voxel_size.tolist(),
        'origin': origin.tolist(),
        'stats': {
            'method': 'percentile_minmax',
            'percentiles': {'low': float(p_low), 'high': float(p_high)},
            'vmin_after_clip': vmin,
            'vmax_after_clip': vmax
        },
        'contour': contour,
        'source': {'map_path': map_path, 'cif_path': cif_path},
        'note': (
            'volume in map.zarr/volume is ModelAngelo-style normalized: '
            'clip to [p1,p99] then min-max to [0,1]. '
            'points.npy computed on RAW (pre-normalization) density.'
        )
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return out_dir, pts.shape[0]



def parse_table(path: str):
    # Detect delimiter (comma or tab)
    with open(path, 'r', newline='') as f:
        head = f.read(1024)
        delim = '	' if ('	' in head and ',' not in head) else ','
    # Read
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter=delim)
        required = {'map_path', 'cif_path', 'contour'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Input table must have headers: {required}. Found: {reader.fieldnames}")
        for r in reader:
            rows.append(r)
    return rows
