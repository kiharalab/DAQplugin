#!/usr/bin/env python3
"""
DAQ-Score: A Deep-learning-based residue-wise Quality Assessment score for cryo-EM models
Map-grid based fast version

Standalone Python script version of DAQ_Score_Grid.ipynb

Usage:
    python daq_score_grid.py --map <map_file.mrc> [--pdb <pdb_file.pdb>] [--output <output_dir>] [--contour <contour>] [--stride <stride>] [--batch_size <batch_size>] [--model <model_path>]

Copyright (C) 2021 Genki Terashi*, Xiao Wang*, Sai Raghavendra Maddhuri Venkata Subramaniya,
John J. G. Tesmer, and Daisuke Kihara, and Purdue University.

License: GPL v3 for academic use.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import tqdm

# Import DAQ modules
import map_util.prep_points_from_mrc as prep
from map_util.dataset_map import MapPointPatchDataset
from map_util.resize_map import resize_map
from map_util.unify_map import Unify_Map
from DAQ.models.resnet import resnet18 as resnet18_multi


class PatchDS(Dataset):
    """Dataset for patch data"""

    def __init__(self, arr):  # arr: (N,1,V,V,V) float32
        self.x = arr

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        v = self.x[i]  # (1,V,V,V)
        return torch.from_numpy(v)


def load_model(ckpt_path, voxel_size, device):
    """Load the DAQ model"""
    model = resnet18_multi(sample_size=voxel_size).to(device)
    model = nn.DataParallel(model)
    # weights_only=False is safe here since we trust the model checkpoint files
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def check_input_files(map_file, pdb_files=None):
    """Check input files and return validated paths"""
    map_path = Path(map_file)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_file}")

    if map_path.suffix.lower() not in {".map", ".mrc"}:
        raise ValueError(f"Map file must be .map or .mrc, got: {map_path.suffix}")

    model_files = []
    if pdb_files:
        for pdb_file in pdb_files:
            pdb_path = Path(pdb_file)
            if not pdb_path.exists():
                raise FileNotFoundError(f"PDB/CIF file not found: {pdb_file}")
            if pdb_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
                raise ValueError(
                    f"Model file must be .pdb, .cif, or .mmcif, got: {pdb_path.suffix}"
                )
            model_files.append(pdb_path)

    return map_path, model_files


def process_map(map_path, output_dir, contour=0.0, stride=2, max_points=500000):
    """Process map: unify, resize and prepare points"""
    map_id = map_path.stem

    # Create output directories
    unified_dir = Path(output_dir) / "unified_map"
    resampled_dir = Path(output_dir) / "resampled_map"
    daqinp_dir = Path(output_dir) / "DAQinp"
    unified_dir.mkdir(parents=True, exist_ok=True)
    resampled_dir.mkdir(parents=True, exist_ok=True)
    daqinp_dir.mkdir(parents=True, exist_ok=True)

    # Unify map
    unified_map_path = unified_dir / f"{map_id}_unified.mrc"
    if not unified_map_path.exists():
        print(f"Unifying map: {map_path} -> {unified_map_path}")
        Unify_Map(str(map_path), str(unified_map_path))

    # Resample map
    new_map_path = resampled_dir / f"{map_id}_resampled.map"
    if not new_map_path.exists():
        print(f"Resizing map: {unified_map_path} -> {new_map_path}")
        resize_map(str(unified_map_path), str(new_map_path))

    # Prepare points
    protein_entry = {
        "pdb_id": "NA",
        "emdb_id": map_id,
        "map_path": str(unified_map_path),
        "cif_path": "NA",
        "contour": float(contour),
    }

    print(f"Preparing points for {map_id}...")
    out_dir, npts = prep.build_one(
        protein_entry, out_root=str(daqinp_dir), max_points=max_points, stride=stride
    )

    print(f"Built {out_dir}  points={npts}")
    return map_id, out_dir, npts


def run_daq_scoring(
    map_id, daqinp_dir, n_points, model_path, output_dir, batch_size=512, device="cuda"
):
    """Run DAQ scoring on the prepared points"""
    print(f"Running DAQ scoring for {map_id}...")

    # Load dataset
    data = MapPointPatchDataset(
        root=str(daqinp_dir), map_ids=[f"{map_id}_resampled"], Np=n_points, patch_vox=11
    )

    sample = data[0]
    patches = sample["patches"]
    points = sample["points"]

    # Process patches
    if isinstance(patches, torch.Tensor):
        patches = patches.cpu()

    V = patches.shape[-1]
    assert V == 11, f"VOXEL_SIZE is not 11. Please check: {V}"

    if patches.ndim == 4:
        patches = patches.unsqueeze(1)  # (N,11,11,11) → (N,1,11,11,11)
    elif patches.ndim == 3:
        patches = patches.unsqueeze(0).unsqueeze(0)  # (11,11,11) → (1,1,11,11,11)
    elif patches.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected patch shape: {patches.shape}")

    patches = patches.to(torch.float32).numpy()

    # Swap XZ axes: (N,1,X,Y,Z) -> (N,1,Z,Y,X)
    SWAP_XZ = True
    if SWAP_XZ:
        patches = np.transpose(patches, (0, 1, 4, 3, 2)).copy()

    # Create DataLoader
    dl = DataLoader(
        PatchDS(patches),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, V, device)

    # Run predictions
    print("Running predictions...")
    pred1_all, pred2_all, pred3_all = [], [], []
    with torch.no_grad():
        for b in tqdm(dl, desc="Predicting"):
            b = b.to(device)  # (B,1,11,11,11)
            p1, p2, p3 = model(b)  # logits
            pred1_all.append(F.softmax(p1, 1).cpu().numpy())
            pred2_all.append(F.softmax(p2, 1).cpu().numpy())
            pred3_all.append(F.softmax(p3, 1).cpu().numpy())

    pred1 = np.concatenate(pred1_all, 0)  # (N,C1)
    pred2 = np.concatenate(pred2_all, 0)  # (N,C2)
    pred3 = np.concatenate(pred3_all, 0)  # (N,C3)

    # Convert points to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    assert points.shape[0] == pred1.shape[0] == pred2.shape[0] == pred3.shape[0], (
        "#points != #prediction"
    )

    # Compute DAQ log scores
    eps = 1e-12
    ref_aa = pred1.mean(axis=0)  # (20,)
    ref_aa = np.clip(ref_aa, eps, 1.0)
    ref_atom = pred2.mean(axis=0)  # (6,)
    ref_atom = np.clip(ref_atom, eps, 1.0)
    ref_ss = pred3.mean(axis=0)  # (3,)
    ref_ss = np.clip(ref_ss, eps, 1.0)

    pred1_logratio = np.log(np.clip(pred1, eps, 1.0) / ref_aa[None, :]).astype(
        np.float32
    )  # (N,20)
    pred2_logratio = np.log(np.clip(pred2, eps, 1.0) / ref_atom[None, :]).astype(
        np.float32
    )  # (N,6)
    pred3_logratio = np.log(np.clip(pred3, eps, 1.0) / ref_ss[None, :]).astype(
        np.float32
    )  # (N,3)

    # Save results
    data_all = np.concatenate(
        [
            points.astype(np.float32),  # (N,3)
            pred1_logratio.astype(np.float32),  # (N,20)
            pred2_logratio.astype(np.float32),  # (N,6)
            pred3_logratio.astype(np.float32),  # (N,3)
        ],
        axis=1,
    )  # (N,32)

    output_file = Path(output_dir) / f"{map_id}_points_AA_ATOM_SS.npy"
    np.save(str(output_file), data_all)
    print(f"Saved: {output_file} {data_all.shape}")

    return output_file, data_all


def main():
    parser = argparse.ArgumentParser(
        description="DAQ-Score: Map-grid based fast version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with map file only
  python daq_score_grid.py --map example.mrc --output ./results
  
  # With PDB file
  python daq_score_grid.py --map example.mrc --pdb model.pdb --output ./results
  
  # With custom parameters
  python daq_score_grid.py --map example.mrc --contour 0.0035 --stride 2 --batch_size 1024 --output ./results
        """,
    )

    parser.add_argument(
        "--map", required=True, type=str, help="Input map file (.mrc or .map)"
    )
    parser.add_argument(
        "--pdb",
        nargs="+",
        type=str,
        default=None,
        help="Optional PDB/CIF file(s) to score",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./OutPuts",
        help="Output directory (default: ./OutPuts)",
    )
    parser.add_argument(
        "--contour",
        type=float,
        default=0.0,
        help="Contour level for the input map (default: 0.0)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride step for scanning the cryo-EM map (default: 2, suggested range: [1,4])",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for prediction (default: 512)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (default: DAQ/best_model/qa_model/Multimodel.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=500000,
        help="Maximum number of points to process (default: 500000)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Set default model path
    if args.model is None:
        script_dir = Path(__file__).parent
        model_path = script_dir / "DAQ" / "best_model" / "qa_model" / "Multimodel.pth"
        if not model_path.exists():
            # Try alternative location
            model_path = script_dir / "DAQ" / "best_model" / "Multimodel.pth"
    else:
        model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check input files
    map_path, model_files = check_input_files(args.map, args.pdb)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process map
    map_id, daqinp_dir, n_points = process_map(
        map_path,
        output_dir,
        contour=args.contour,
        stride=args.stride,
        max_points=args.max_points,
    )

    # Run DAQ scoring
    output_file, data_all = run_daq_scoring(
        map_id,
        Path(
            daqinp_dir
        ).parent,  # DAQinp directory (parent of the specific map directory)
        n_points,
        str(model_path),
        output_dir,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"\nDAQ scoring completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Output shape: {data_all.shape}")
    print(f"Output directory: {output_dir}")

    # Score PDB files if provided
    if model_files:
        score_pdb_files(model_files, output_file, output_dir)


def score_pdb_files(model_files, points_file, output_dir):
    """Score PDB/CIF files using the computed DAQ scores"""
    import subprocess

    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if daq_write_bfactor.py exists
    script_dir = Path(__file__).parent
    daq_script = script_dir / "cli" / "daq_write_bfactor.py"

    if not daq_script.exists():
        print(f"\nWarning: {daq_script} not found. Skipping PDB scoring.")
        print(f"PDB files provided: {[str(f) for f in model_files]}")
        return

    scored_list = []
    for pdb_file in model_files:
        print(f"\nComputing DAQ(AA) score for {pdb_file}")
        pdb_path = Path(pdb_file)
        model_id = pdb_path.stem
        output_model = models_dir / f"{model_id}_daq{pdb_path.suffix}"

        # Call daq_write_bfactor.py
        cmd = [
            sys.executable,
            str(daq_script),
            "-i",
            str(pdb_path),
            "-o",
            str(output_model),
            "-p",
            str(points_file),
            "-m",
            "aa_score",
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            scored_list.append(output_model)
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {pdb_file}: {e}")
            print(f"Error output: {e.stderr}")

    if scored_list:
        print(f"\nSuccessfully scored {len(scored_list)} model(s):")
        for f in scored_list:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
