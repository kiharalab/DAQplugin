"""
MLX backend for DAQ inference on Apple Silicon.

Why MLX: ONNX Runtime CoreML EP only supports 1D/2D Conv. The DAQ network is a
3D ResNet-18 (3D Conv over 11x11x11 patches), so ORT falls back entirely to CPU
on Mac. ORT has no MPS provider. MLX runs natively on Apple GPU via Metal and
supports 3D conv, giving a real GPU path on M-series Macs.

Model: resnet18_multi(sample_size=11) = ResNet_custom(Bottleneck, [1,2,2,1]),
3 heads -> AA(20), atom(6), SS(3) logits. Softmax applied here to match ORT path.

Layout: PyTorch uses NCDHW; MLX uses NDHWC. Inputs are transposed at the
boundary; weights are pre-transposed by tools/convert_pth_to_mlx.py.
"""
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _is_mlx_oom(exc: BaseException) -> bool:
    """Detect Metal/CUDA OOM coming out of MLX."""
    msg = (str(exc) or "").lower()
    return any(n in msg for n in (
        "out of memory",
        "[metal::malloc]",
        "[conv] cached plan failed to execute",  # often follows OOM in MLX-CUDA
        "memoryerror",
    ))


def _import_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        return mx, nn
    except ImportError as e:
        raise ImportError(
            "MLX is not installed. On macOS, install with:\n"
            "    pip install mlx\n"
            f"(import error: {e})"
        )


def _build_resnet18_multi(sample_size: int = 11):
    """Construct the MLX ResNet_custom equivalent of resnet18_multi.

    Mirrors DAQ/models/resnet.py: ResNet_custom(Bottleneck, [1,2,2,1],
    sample_size=11, cardinality=32, num_classes=[20,6,3]).
    """
    mx, nn = _import_mlx()
    cardinality = 32
    num_classes = (20, 6, 3)
    layers_cfg = [1, 2, 2, 1]
    expansion = 2

    class BN3d(nn.Module):
        """BatchNorm3d for NDHWC input. mlx.nn.BatchNorm only accepts rank
        2/3/4 inputs, so we apply the affine + running-stat formula directly
        along the last axis. Parameter attribute names match nn.BatchNorm so
        the .npz keys (bnX.weight, bnX.bias, bnX.running_mean, bnX.running_var)
        load without remapping.
        """

        def __init__(self, num_features, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))

        def __call__(self, x):
            # Inference path only: use frozen running statistics. Broadcasting
            # over leading dims (N, D, H, W, C) -> per-channel C.
            normed = (x - self.running_mean) * mx.rsqrt(self.running_var + self.eps)
            return normed * self.weight + self.bias

    class GroupedConv3d(nn.Module):
        """Grouped 3D convolution. MLX (>=0.31) does not yet support groups>1
        for 3D convs (mx.conv3d / mx.conv_general both raise). Implement manually
        by splitting along the channel axis.

        Weight layout matches an ungrouped Conv3d:
            shape = (C_out, kD, kH, kW, C_in // groups)
        which is also what tools/convert_pth_to_mlx.py emits, so the existing
        .npz keys load directly with no special-case in the converter.
        """

        def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
            super().__init__()
            assert in_ch % groups == 0 and out_ch % groups == 0
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.groups = groups
            self.stride = stride
            self.padding = padding
            kD = kH = kW = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            self.kernel_size = (kD, kH, kW)
            # Single weight tensor for entire layer; sliced per-group at fwd time.
            self.weight = mx.zeros((out_ch, kD, kH, kW, in_ch // groups), dtype=mx.float32)

        def __call__(self, x):
            # x: NDHWC, C_in == self.in_ch
            ich_per_g = self.in_ch // self.groups
            och_per_g = self.out_ch // self.groups
            outs = []
            for g in range(self.groups):
                xg = x[..., g * ich_per_g:(g + 1) * ich_per_g]
                wg = self.weight[g * och_per_g:(g + 1) * och_per_g]
                outs.append(mx.conv_general(
                    xg, wg, stride=self.stride, padding=self.padding, groups=1,
                ))
            return mx.concatenate(outs, axis=-1)

    class Bottleneck(nn.Module):
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            mid = cardinality * (planes // 32)
            self.conv1 = nn.Conv3d(inplanes, mid, kernel_size=1, bias=False)
            self.bn1 = BN3d(mid)
            self.conv2 = GroupedConv3d(
                mid, mid, kernel_size=3, stride=stride, padding=1, groups=cardinality,
            )
            self.bn2 = BN3d(mid)
            self.conv3 = nn.Conv3d(
                mid, planes * expansion, kernel_size=1, bias=False,
            )
            self.bn3 = BN3d(planes * expansion)
            self.downsample = downsample

        def __call__(self, x):
            residual = x
            out = nn.relu(self.bn1(self.conv1(x)))
            out = nn.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                residual = self.downsample(x)
            return nn.relu(out + residual)

    class ResNetCustom(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = BN3d(64)
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(128, layers_cfg[0])
            self.layer2 = self._make_layer(256, layers_cfg[1], stride=2)
            self.layer3 = self._make_layer(512, layers_cfg[2], stride=2)
            self.layer4 = self._make_layer(1024, layers_cfg[3], stride=2)
            last = int(math.ceil(sample_size / 32))
            # PyTorch uses AvgPool3d((last,last,last), stride=1). For
            # sample_size=11, math.ceil(11/32)=1 -> 1x1x1 pool (identity).
            self.avgpool = nn.AvgPool3d(kernel_size=last, stride=1)
            feat = cardinality * 32 * expansion
            self.fc1 = nn.Linear(feat, num_classes[0])
            self.fc2 = nn.Linear(feat, num_classes[1])
            self.fc3 = nn.Linear(feat, num_classes[2])

        def _make_layer(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * expansion, kernel_size=1,
                              stride=stride, bias=False),
                    BN3d(planes * expansion),
                )
            mods = [Bottleneck(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * expansion
            for _ in range(1, blocks):
                mods.append(Bottleneck(self.inplanes, planes))
            return nn.Sequential(*mods)

        def __call__(self, x):
            x = nn.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            return self.fc1(x), self.fc2(x), self.fc3(x)

    return ResNetCustom()


def _softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class DAQMLXModel:
    """MLX-backed inference wrapper matching the DAQOnnxModel API surface used
    by compute.py: predict_batched(patches, batch_size, progress_callback).
    """

    def __init__(self, weights_path: str, verbose: bool = False,
                 sample_size: int = 11, device: str = "gpu"):
        """
        Parameters
        ----------
        device : str
            'gpu' = Metal (Apple Silicon default). 'cpu' = MLX CPU device
            (Intel Mac, or when user opts out of GPU). Falls back to 'cpu' if
            Metal isn't available on this host.
        """
        mx, nn = _import_mlx()
        self.mx = mx
        self.nn = nn
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"MLX weights not found: {self.weights_path}")

        # Resolve target device. mx.stream(mx.gpu) raises on non-Metal builds;
        # detect by attempting a small op under that context and fall back to
        # CPU. Stored mx.Device object is used in mx.stream(...) per call so
        # two coexisting models (one GPU, one CPU) cannot stomp on each other
        # via the process-global default device.
        requested = (device or "gpu").lower()
        if requested == "gpu":
            try:
                with mx.stream(mx.gpu):
                    _probe = mx.array([0.0], dtype=mx.float32)
                    mx.eval(_probe)
                self._device_obj = mx.gpu
                self.device = "gpu"
            except Exception:
                self._device_obj = mx.cpu
                self.device = "cpu"
                print("DAQ: MLX Metal unavailable, using MLX CPU device")
        else:
            self._device_obj = mx.cpu
            self.device = "cpu"

        self.provider = f"MLX-{self.device}"
        self.gpu_id = 0
        self.input_name = "input"  # for API compatibility
        self.input_shape = [None, 1, sample_size, sample_size, sample_size]
        self.output_names = ["aa", "atom", "ss"]
        self.sample_size = sample_size

        # Build + load + eval inside the device scope so weights, BN running
        # stats, and Conv kernels are materialized on the target device.
        with mx.stream(self._device_obj):
            self.model = _build_resnet18_multi(sample_size=sample_size)
            # Load pre-converted weights (.npz from tools/convert_pth_to_mlx.py).
            self.model.load_weights(str(self.weights_path), strict=True)
            # Eval mode (disable Dropout / BatchNorm running-stats updates).
            self.model.eval()
            # Force materialization on the target device.
            mx.eval(self.model.parameters())
        backend_label = "Metal" if self.device == "gpu" else "CPU"
        print(f"DAQ: MLX backend ready ({backend_label}); "
              f"weights={self.weights_path.name}")

    def __repr__(self) -> str:
        return f"DAQMLXModel(weights={self.weights_path.name}, provider=MLX)"

    def predict(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on one batch.

        Parameters
        ----------
        patches : np.ndarray
            Input patches with shape (N, 1, D, H, W) NCDHW float32 -- same
            layout the ORT path receives.

        Returns
        -------
        (aa_probs, atom_probs, ss_probs) : tuple of np.ndarray
            Softmax probabilities. aa: (N,20), atom: (N,6), ss: (N,3).
        """
        mx = self.mx
        if patches.dtype != np.float32:
            patches = patches.astype(np.float32)
        if patches.ndim == 4:
            patches = patches[:, np.newaxis, :, :, :]
        # NCDHW -> NDHWC for MLX
        patches_ndhwc = np.ascontiguousarray(np.transpose(patches, (0, 2, 3, 4, 1)))
        # Pin every op (input array creation, forward pass, eval) to the
        # device this model was built on, regardless of the current global
        # default. Without this, another DAQMLXModel created on a different
        # device would silently steal compute.
        with mx.stream(self._device_obj):
            x = mx.array(patches_ndhwc)
            aa_logits, atom_logits, ss_logits = self.model(x)
            mx.eval(aa_logits, atom_logits, ss_logits)
            aa = np.asarray(aa_logits, dtype=np.float32)
            atom = np.asarray(atom_logits, dtype=np.float32)
            ss = np.asarray(ss_logits, dtype=np.float32)
        return _softmax_np(aa, axis=1), _softmax_np(atom, axis=1), _softmax_np(ss, axis=1)

    def get_optimal_batch_size(self, n_patches: int, patch_shape: tuple = (1, 11, 11, 11)) -> int:
        # Throughput cap from tools/profile_batch_size.py (RTX 6000 Ada via
        # mlx-cuda — Mac Metal expected to follow similar curve).
        # MLX-CUDA scaled near-linearly to batch 2048 (~75K p/s) and topped
        # out at batch 4096 (~85K p/s). Batch 4096 intermittently failed
        # with `[conv] Cached plan failed to execute`, so cap at 2048 for
        # safety. CPU device falls back to same number; not throughput
        # competitive but keeps logic simple.
        cap = 2048
        batch_size = min(cap, max(1, n_patches))
        backend = "Metal" if self.device == "gpu" else "CPU"
        print(f"DAQ: Auto batch size: {batch_size} (MLX/{backend}, cap {cap})")
        return batch_size

    def predict_batched(
        self,
        patches: np.ndarray,
        batch_size: int = 0,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = patches.shape[0]
        if batch_size is None or batch_size <= 0:
            batch_size = self.get_optimal_batch_size(N, patches.shape[1:])

        # OOM retry: halve batch on Metal/CUDA OOM until batch=1 or success.
        original_batch = batch_size
        while batch_size >= 1:
            try:
                return self._predict_loop(patches, batch_size, progress_callback)
            except (RuntimeError, MemoryError) as exc:
                if not _is_mlx_oom(exc):
                    raise
                if batch_size == 1:
                    raise RuntimeError(
                        f"DAQ: MLX OOM even at batch_size=1; cannot recover. "
                        f"Original error: {exc}") from exc
                new_batch = max(1, batch_size // 2)
                print(f"DAQ: MLX OOM at batch={batch_size}, retrying at "
                      f"batch={new_batch} (started at {original_batch})")
                batch_size = new_batch

    def _predict_loop(self, patches, batch_size, progress_callback):
        N = patches.shape[0]
        aa_all, atom_all, ss_all = [], [], []
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch = patches[i:end]
            aa, atom, ss = self.predict(batch)
            aa_all.append(aa)
            atom_all.append(atom)
            ss_all.append(ss)
            if progress_callback is not None:
                progress_callback(end, N)
        return (
            np.concatenate(aa_all, axis=0),
            np.concatenate(atom_all, axis=0),
            np.concatenate(ss_all, axis=0),
        )


# ---------- weights file path ----------

MLX_WEIGHTS_FILENAME = "Multimodel.mlx.npz"
MLX_WEIGHTS_URL = (
    "https://huggingface.co/zhtronics/DAQscore/resolve/main/Multimodel.mlx.npz"
)


def download_mlx_weights(dest_path: Path, url: str = MLX_WEIGHTS_URL) -> bool:
    """Download the converted MLX weights to dest_path. Mirrors
    onnx_model.download_model() so behaviour and progress reporting match.
    """
    import sys
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DAQ MLX weights from {url}...")
    print(f"Destination: {dest_path}")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nMLX weights download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def get_mlx_weights_path(auto_download: bool = True) -> Optional[Path]:
    """Locate the converted MLX weights, downloading from Hugging Face on
    first use if missing.

    Search order mirrors get_model_path() for the ONNX file:
      1. DAQ_MLX_WEIGHTS env var
      2. Plugin data/ directory (installed)
      3. Plugin data/ directory (development)
      4. ~/.chimerax/daq_model/Multimodel.mlx.npz  (also download target)
    """
    import os
    candidates = []
    env_path = os.environ.get("DAQ_MLX_WEIGHTS")
    if env_path:
        candidates.append(Path(env_path))
    module_dir = Path(__file__).parent
    candidates.append(module_dir / "data" / MLX_WEIGHTS_FILENAME)
    candidates.append(module_dir.parent / "data" / MLX_WEIGHTS_FILENAME)
    user_path = Path.home() / ".chimerax" / "daq_model" / MLX_WEIGHTS_FILENAME
    candidates.append(user_path)
    for p in candidates:
        if p.exists():
            return p

    if auto_download:
        print("DAQ: MLX weights not found locally; downloading from Hugging Face.")
        if download_mlx_weights(user_path):
            return user_path

    return None
