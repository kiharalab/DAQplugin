#!/usr/bin/env python3
"""
Linux test harness for the MLX backend.

Compares MLX inference output against the PyTorch reference using the
shipped Multimodel.mlx.npz weights. Lets us catch MLX-port bugs without a
Mac.

Setup:
    python -m venv /tmp/venv
    /tmp/venv/bin/pip install torch numpy 'mlx[cpu]'

Run:
    /tmp/venv/bin/python tools/test_mlx_parity.py

If your host's g++ rejects MLX's JIT-compiled CPU kernels (some toolchains
have a `_Float128` redeclaration warning), set:
    export MLX_DISABLE_COMPILE=1
or call ``mx.disable_compile()`` before the first ``mx.eval``. The model
runs in MLX's interpreted CPU path; outputs are bit-equivalent to the
compiled path. Mac MLX (Metal) needs no g++ — kernels ship precompiled.

Pass criterion: max-abs softmax diff < 5e-3 (float32 FMA reordering noise
is ~1e-6 in practice).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO / "daqcolor" / "src" / "data" / "Multimodel.mlx.npz"
DEFAULT_PTH = REPO / "DAQ" / "best_model" / "qa_model" / "Multimodel.pth"
DEFAULT_ONNX = Path.home() / ".chimerax" / "daq_model" / "Multimodel.onnx"


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--pth", default=str(DEFAULT_PTH))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=5e-3)
    parser.add_argument("--no-onnx", action="store_true",
                        help="Skip ONNX comparison (use only PyTorch reference)")
    parser.add_argument("--no-pth", action="store_true",
                        help="Skip PyTorch comparison (use only ONNX reference)")
    parser.add_argument("--via-wrapper", action="store_true",
                        help="Run inference through the DAQMLXModel.predict_batched "
                             "API path instead of raw _build_resnet18_multi() — "
                             "exercises the same code path compute.py uses.")
    args = parser.parse_args()

    if os.environ.get("MLX_DISABLE_COMPILE", "0") == "1":
        import mlx.core as _mx_pre
        _mx_pre.disable_compile()

    sys.path.insert(0, str(REPO / "DAQ"))
    sys.path.insert(0, str(REPO / "daqcolor" / "src"))

    import mlx.core as mx
    import mlx_model

    print("Building MLX model...")
    if args.via_wrapper:
        # Goes through the same path compute.py uses on Mac:
        # NCDHW input -> internal transpose -> model -> softmax_np.
        wrapper = mlx_model.DAQMLXModel(args.weights)
    else:
        model_mlx = mlx_model._build_resnet18_multi(sample_size=11)
        model_mlx.load_weights(args.weights, strict=True)
        model_mlx.eval()
        mx.eval(model_mlx.parameters())

    rng = np.random.default_rng(args.seed)
    batch = rng.standard_normal((args.batch, 1, 11, 11, 11)).astype(np.float32)

    print("Running MLX inference...")
    if args.via_wrapper:
        aa_mlx, atom_mlx, ss_mlx = wrapper.predict_batched(batch, batch_size=args.batch)
    else:
        x_mlx = mx.array(np.transpose(batch, (0, 2, 3, 4, 1)))
        aa_l, atom_l, ss_l = model_mlx(x_mlx)
        mx.eval(aa_l, atom_l, ss_l)
        aa_mlx = softmax(np.asarray(aa_l), 1)
        atom_mlx = softmax(np.asarray(atom_l), 1)
        ss_mlx = softmax(np.asarray(ss_l), 1)

    references = []

    if not args.no_pth and Path(args.pth).exists():
        try:
            import torch
            from models.resnet import resnet18 as resnet18_multi
            print("Running PyTorch reference...")
            model_pt = resnet18_multi(sample_size=11)
            sd = torch.load(args.pth, map_location="cpu", weights_only=False)["state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model_pt.load_state_dict(sd)
            model_pt.eval()
            with torch.no_grad():
                aa_pt_l, atom_pt_l, ss_pt_l = model_pt(torch.from_numpy(batch))
            references.append(("PyTorch",
                               softmax(aa_pt_l.numpy(), 1),
                               softmax(atom_pt_l.numpy(), 1),
                               softmax(ss_pt_l.numpy(), 1)))
        except ImportError:
            print("  (torch not installed — skipping PyTorch reference)")

    if not args.no_onnx and Path(args.onnx).exists():
        try:
            import onnxruntime as ort
            print(f"Running ONNX reference ({args.onnx})...")
            sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            outs = sess.run(None, {in_name: batch})
            # ONNX exports raw logits in the same order: aa(20), atom(6), ss(3).
            # _build_resnet18_multi returns logits too, so apply softmax for parity.
            references.append(("ONNX",
                               softmax(outs[0], 1),
                               softmax(outs[1], 1),
                               softmax(outs[2], 1)))
        except ImportError:
            print("  (onnxruntime not installed — skipping ONNX reference)")
    elif not args.no_onnx:
        print(f"  (ONNX file not found at {args.onnx} — skipping)")

    if not references:
        print("ERROR: no reference backends available; cannot verify parity.", file=sys.stderr)
        return 2

    failed = False
    for ref_name, aa_ref, atom_ref, ss_ref in references:
        print(f"\n=== MLX vs {ref_name} ===")
        for head, m, r in [("aa", aa_mlx, aa_ref),
                           ("atom", atom_mlx, atom_ref),
                           ("ss", ss_mlx, ss_ref)]:
            diff = np.abs(m - r)
            ok = diff.max() < args.tol
            status = "PASS" if ok else "FAIL"
            print(f"  {head:5s} shape={tuple(m.shape)} max_diff={diff.max():.2e} "
                  f"mean_diff={diff.mean():.2e} {status}")
            if not ok:
                failed = True
                print(f"    MLX[0]={m[0]}")
                print(f"    {ref_name}[0]={r[0]}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
