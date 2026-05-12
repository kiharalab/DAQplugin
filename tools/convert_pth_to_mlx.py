#!/usr/bin/env python3
"""
One-time conversion: DAQ PyTorch checkpoint (.pth) -> MLX weights (.npz).

The runtime MLX backend (daqcolor/src/mlx_model.py) loads the .npz at startup
and never imports torch. We do the conversion offline here so that PyTorch is
NOT a runtime dependency on Mac.

Conversion handles two boundaries between PyTorch and MLX:
  1) DataParallel prefix: state_dict keys begin with "module.". Strip it.
  2) Conv3d weight layout: PyTorch (out, in/groups, kD, kH, kW) -> MLX
     (out, kD, kH, kW, in/groups). MLX uses NDHWC layout; weights match.
  3) Sequential children path: PyTorch nn.Sequential indexes children directly
     ("layer1.0.conv1.weight"); MLX nn.Sequential exposes them under .layers
     ("layer1.layers.0.conv1.weight"). Insert "layers." after each
     {layer1,layer2,layer3,layer4,downsample} segment that's followed by an
     integer index.
  4) BatchNorm: drop num_batches_tracked (PyTorch bookkeeping not in MLX).

Usage:
    python tools/convert_pth_to_mlx.py \\
        DAQ/best_model/qa_model/Multimodel.pth \\
        daqcolor/data/Multimodel.mlx.npz

After conversion, the runtime resolves the .npz via:
  $DAQ_MLX_WEIGHTS, daqcolor/data/, ~/.chimerax/daq_model/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


# Names of attributes in our MLX ResNet that wrap nn.Sequential. PyTorch
# state_dict keys index Sequential children directly; MLX nests under .layers.
_SEQUENTIAL_PARENTS = ("layer1", "layer2", "layer3", "layer4", "downsample")


def _remap_key(pt_key: str) -> str:
    # 1) Strip DataParallel prefix.
    if pt_key.startswith("module."):
        pt_key = pt_key[len("module."):]

    # 2) Insert ".layers" after each Sequential parent followed by an int index.
    #    e.g. "layer1.0.conv1.weight" -> "layer1.layers.0.conv1.weight"
    #         "layer1.0.downsample.0.weight" ->
    #         "layer1.layers.0.downsample.layers.0.weight"
    parts = pt_key.split(".")
    out = []
    i = 0
    while i < len(parts):
        out.append(parts[i])
        if parts[i] in _SEQUENTIAL_PARENTS and i + 1 < len(parts) and parts[i + 1].isdigit():
            out.append("layers")
        i += 1
    return ".".join(out)


def _convert_value(key: str, arr: np.ndarray) -> np.ndarray:
    # Conv3d weight: (out, in/groups, kD, kH, kW) -> (out, kD, kH, kW, in/groups)
    if arr.ndim == 5 and key.endswith(".weight"):
        return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 4, 1)))
    return np.ascontiguousarray(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DAQ PyTorch checkpoint to MLX npz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pth_path", type=Path, help="Input .pth checkpoint")
    parser.add_argument("npz_path", type=Path, help="Output .npz weights file")
    parser.add_argument("--print-keys", action="store_true",
                        help="Print key remapping table to stderr")
    args = parser.parse_args()

    if not args.pth_path.exists():
        print(f"ERROR: {args.pth_path} not found", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required ONLY for this offline converter.\n"
              "       pip install torch", file=sys.stderr)
        return 2

    print(f"Loading {args.pth_path} ...", file=sys.stderr)
    ckpt = torch.load(args.pth_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    out = {}
    skipped = []
    for pt_key, val in state.items():
        if pt_key.endswith("num_batches_tracked"):
            skipped.append(pt_key)
            continue
        arr = val.detach().cpu().numpy()
        mlx_key = _remap_key(pt_key)
        out[mlx_key] = _convert_value(mlx_key, arr)
        if args.print_keys:
            print(f"  {pt_key:60s} -> {mlx_key:70s} {out[mlx_key].shape} {out[mlx_key].dtype}",
                  file=sys.stderr)

    args.npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.npz_path, **out)
    total_bytes = sum(v.nbytes for v in out.values())
    print(f"Wrote {args.npz_path} ({len(out)} arrays, "
          f"{total_bytes / 1024 / 1024:.2f} MB)", file=sys.stderr)
    if skipped:
        print(f"Skipped {len(skipped)} num_batches_tracked entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
