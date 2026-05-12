# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ONNX Runtime inference wrapper for DAQ model.

Backend is the single source of truth for which inference path runs.
Canonical values (passed via `backend=` everywhere):

    auto      Platform fallback chain (see _auto_chain).
    tensorrt  ORT TensorRT EP (Linux/Windows, NVIDIA).
    cuda      ORT CUDA EP (Linux, NVIDIA).
    directml  ORT DirectML EP (Windows, AMD/Intel/NVIDIA).
    mlx       MLX Metal device (macOS Apple Silicon).
    mlx-cpu   MLX CPU device (macOS Accelerate + AMX; faster than ORT CPU
              on Apple Silicon, much slower on Linux).
    cpu       ORT CPUExecutionProvider on any platform.

Forced backends raise on unavailability so the user knows their choice
can't be honored. `auto` walks the chain and returns the first that loads.
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Callable

# Import platform detection from constants
try:
    from .constants import PLATFORM, MODEL_URL, MODEL_FILENAME
except ImportError:
    # Fallback for standalone use
    PLATFORM = 'linux' if sys.platform.startswith('linux') else ('darwin' if sys.platform == 'darwin' else 'windows')
    MODEL_URL = "https://huggingface.co/zhtronics/DAQscore/resolve/main/Multimodel.onnx"
    MODEL_FILENAME = "Multimodel.onnx"

# Global model cache to avoid reloading the model on each computation.
# Key: (resolved_backend, gpu_id, model_path_str). resolved_backend is the
# concrete choice (never "auto"), so identical effective configs collapse
# into one cache slot regardless of whether the user asked for them
# explicitly or via `auto`. Cache size is capped at 1 to bound VRAM (see
# _evict_other_cache_entries).
_model_cache = {}

# Canonical backend names. "auto" is virtual — gets resolved before dispatch.
VALID_BACKENDS = frozenset({
    "auto", "tensorrt", "cuda", "directml", "mlx", "mlx-cpu", "cpu",
})
_ORT_BACKENDS = frozenset({"tensorrt", "cuda", "directml", "cpu"})
_MLX_BACKENDS = frozenset({"mlx", "mlx-cpu"})


def _auto_chain():
    """Ordered list of backends to try for backend='auto', per platform.

    Linux:   tensorrt -> cuda -> cpu
    Windows: tensorrt -> directml -> cpu
    macOS:   mlx -> mlx-cpu -> cpu
        (Apple Silicon: MLX CPU uses Accelerate + AMX coprocessor, much
        faster than ORT CPU EP for 3D conv — so it earns its slot in
        the chain. On Intel Macs MLX CPU still beats ORT CPU for our
        workload.)
    """
    if PLATFORM == "darwin":
        return ["mlx", "mlx-cpu", "cpu"]
    if PLATFORM == "windows":
        return ["tensorrt", "directml", "cpu"]
    return ["tensorrt", "cuda", "cpu"]


def _ep_for_backend(backend: str) -> str:
    """Map an ORT backend name to its ExecutionProvider string."""
    return {
        "tensorrt": "TensorrtExecutionProvider",
        "cuda":     "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
        "cpu":      "CPUExecutionProvider",
    }[backend]


def _is_oom_error(exc: BaseException) -> bool:
    """Detect CUDA/MLX/system OOM from exception text.

    ORT, TRT, and MLX all raise RuntimeError with different message
    formats on out-of-memory. We match common substrings rather than
    rely on a specific exception type.
    """
    msg = (str(exc) or "").lower()
    needles = (
        "out of memory",
        "cudaerrormemoryallocation",
        "cuda failure 2",        # ORT-CUDA OOM
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "failed to allocate memory",
        "memoryerror",
    )
    return any(n in msg for n in needles)


def _query_gpu_info(gpu_id: int = 0):
    """Return (name, sm_int) for the given physical GPU, or (None, None).

    sm_int is the compute capability as int: '8.9' -> 89, '7.5' -> 75.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap",
             "--format=csv,noheader",
             "-i", str(gpu_id)],
            text=True, timeout=5,
        )
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        if len(parts) >= 2:
            name = parts[0]
            try:
                sm = int(parts[1].replace(".", ""))
            except ValueError:
                sm = None
            return name, sm
    except Exception:
        pass
    return None, None


def _query_free_gpu_mb(gpu_id: int = 0) -> Optional[int]:
    """Query free memory in MiB on a given physical GPU via nvidia-smi.

    Returns None if nvidia-smi is unavailable or the query fails. Used to
    scale the auto batch cap down on smaller GPUs (8-12 GB consumer cards)
    without hard-coding a per-card formula.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits",
             "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _trt_provider_options(gpu_id: int) -> dict:
    """
    Build TensorRT EP options with persistent engine + timing cache.

    First inference at a new (model, shape, GPU) triple builds a TRT engine
    (seconds to tens of seconds) and caches it on disk. Subsequent process
    starts at the same shape reuse the cached engine in <100 ms. The timing
    cache shares kernel-tactic measurements across shapes, so even a new
    batch size builds faster after the first one.

    Cache location: ~/.chimerax/daq_model/trt_cache/. Stored per-user, not
    bundled with the wheel since engines are GPU-specific (sm_xx targeted).
    """
    cache_dir = Path.home() / ".chimerax" / "daq_model" / "trt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 1 GiB workspace. Lower via DAQ_TRT_WORKSPACE_MB if GPU is shared
    # and autotuner OOM warnings spam the log; model is tiny enough that
    # 256 MiB also works.
    workspace_mb = int(os.environ.get("DAQ_TRT_WORKSPACE_MB", "1024"))
    # Note: trt_log_severity_level (ORT 1.20+) was tried here to suppress
    # OOM tactic-skip warnings, but older ORT rejects the unknown key with
    # "Invalid TensorRT EP option" and falls all the way back to CPU EP --
    # catastrophic 100x slowdown. Workspace cap is the real fix for OOM;
    # tactic-skip warnings remain cosmetic.
    return {
        "device_id": gpu_id,
        "trt_max_workspace_size": workspace_mb * 1024 * 1024,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir),
    }


def _get_cuda_library_paths():
    """
    Find CUDA library paths from pip-installed nvidia packages AND system CUDA.

    Returns
    -------
    list of str
        List of paths containing CUDA libraries
    """
    import site

    cuda_paths = []

    # 1. Check pip-installed nvidia packages (site-packages/nvidia/*/lib)
    site_packages = site.getsitepackages() or []

    user_site = site.getusersitepackages()
    if user_site and user_site not in site_packages:
        site_packages.append(user_site)

    prefix_path = Path(sys.prefix)
    if sys.platform == 'win32':
        prefix_site_packages = prefix_path / 'Lib' / 'site-packages'
    else:
        prefix_site_packages = prefix_path / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'

    if prefix_site_packages.exists() and str(prefix_site_packages) not in site_packages:
        site_packages.append(str(prefix_site_packages))

    for sp in site_packages:
        sp_path = Path(sp)
        nvidia_libs = [
            'cuda_runtime', 'cudnn', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse'
        ]
        for lib in nvidia_libs:
            # Linux: lib directory
            lib_path = sp_path / 'nvidia' / lib / 'lib'
            if lib_path.exists():
                cuda_paths.append(str(lib_path))
            # Windows: bin directory
            bin_path = sp_path / 'nvidia' / lib / 'bin'
            if bin_path.exists():
                cuda_paths.append(str(bin_path))

        # TensorRT pip wheels (tensorrt-cu12) drop libs directly under
        # site-packages/tensorrt_libs/. ORT-TRT EP dlopens libnvinfer.so.10
        # from LD_LIBRARY_PATH; without preloading it falls back to CUDA EP.
        trt_libs = sp_path / 'tensorrt_libs'
        if trt_libs.exists():
            cuda_paths.append(str(trt_libs))

    # 2. Check system CUDA paths (for linux-cuda option or system-installed CUDA)
    system_cuda_paths = [
        # Standard CUDA Toolkit installation
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        # CUDA version-specific paths
        '/usr/local/cuda-11/lib64',
        '/usr/local/cuda-12/lib64',
        # Ubuntu/Debian system packages
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
        # cuDNN paths
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu/libcudnn*',
    ]

    # Check CUDA_HOME / CUDA_PATH environment variables
    for env_var in ['CUDA_HOME', 'CUDA_PATH']:
        cuda_env = os.environ.get(env_var)
        if cuda_env:
            system_cuda_paths.insert(0, os.path.join(cuda_env, 'lib64'))
            system_cuda_paths.insert(0, os.path.join(cuda_env, 'lib'))

    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        for p in ld_path.split(os.pathsep):
            if p and p not in system_cuda_paths:
                system_cuda_paths.append(p)

    # Add existing system paths
    for path in system_cuda_paths:
        if '*' not in path and os.path.isdir(path) and path not in cuda_paths:
            cuda_paths.append(path)

    return cuda_paths


_cuda_preload_done = False
_cuda_preload_success = False


def _preload_cuda_libraries(verbose: bool = False):
    """
    Preload CUDA libraries using ctypes before ONNX Runtime tries to load them.

    On Linux, setting LD_LIBRARY_PATH after Python starts doesn't help libraries
    that are loaded via dlopen from already-loaded shared libraries. We need to
    use ctypes.CDLL to preload the libraries into the process.

    This function is safe to call multiple times - it will only preload once.
    On Windows (DirectML) and macOS (CPU/MLX), this function does nothing.

    Parameters
    ----------
    verbose : bool
        If True, print debug information

    Returns
    -------
    bool
        True if CUDA libraries were successfully preloaded (or not needed)
    """
    global _cuda_preload_done, _cuda_preload_success

    # Only run once
    if _cuda_preload_done:
        return _cuda_preload_success

    _cuda_preload_done = True

    # Skip on macOS (uses CPU/MLX) and Windows (uses DirectML)
    if PLATFORM == 'darwin':
        if verbose:
            print("DAQ: macOS detected - skipping CUDA preload")
        _cuda_preload_success = True
        return True

    if PLATFORM == 'windows':
        if verbose:
            print("DAQ: Windows detected - skipping CUDA preload (will use DirectML)")
        _cuda_preload_success = True
        return True

    import ctypes
    import glob

    cuda_paths = _get_cuda_library_paths()
    if not cuda_paths:
        if verbose:
            print("DAQ: No CUDA library paths found, relying on system CUDA")
        # Not a failure - system CUDA might be in standard paths
        _cuda_preload_success = True
        return True

    if verbose:
        print(f"DAQ: Found {len(cuda_paths)} CUDA library paths")

    # Update LD_LIBRARY_PATH for any future library loads
    if sys.platform != 'win32':
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = os.pathsep.join(cuda_paths)
        if new_paths not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = (
                new_paths + os.pathsep + current_ld_path if current_ld_path
                else new_paths
            )

    # Libraries to preload in dependency order
    # Order matters: dependencies must be loaded first
    lib_patterns = [
        # CUDA runtime (base dependency)
        'libcudart.so*',
        # cuBLAS and cuBLASLt
        'libcublasLt.so*',
        'libcublas.so*',
        # cuDNN component libraries (must be loaded before main libcudnn.so)
        'libcudnn_ops_infer.so*',
        'libcudnn_ops_train.so*',
        'libcudnn_cnn_infer.so*',
        'libcudnn_cnn_train.so*',
        'libcudnn_adv_infer.so*',
        'libcudnn_adv_train.so*',
        # Main cuDNN library (depends on component libraries above)
        'libcudnn.so*',
        # Other libraries
        'libcufft.so*',
        'libcurand.so*',
        'libcusolver.so*',
        'libcusparse.so*',
        # TensorRT (loaded after CUDA + cuDNN since TRT depends on them).
        # Order matters: nvinfer_plugin and nvonnxparser depend on libnvinfer.
        'libnvinfer.so*',
        'libnvinfer_plugin.so*',
        'libnvinfer_lean.so*',
        'libnvinfer_dispatch.so*',
        'libnvonnxparser.so*',
    ]

    loaded_libs = []

    for pattern in lib_patterns:
        loaded_this_pattern = False
        for cuda_path in cuda_paths:
            if loaded_this_pattern:
                break  # outer break: one cuda_path per pattern is enough
            matches = glob.glob(os.path.join(cuda_path, pattern))
            # Sort to get the most specific version first (e.g., libcublas.so.11 before libcublas.so)
            matches.sort(key=lambda x: len(x), reverse=True)

            for lib_path in matches:
                # Skip symlinks to avoid loading the same library twice
                if os.path.islink(lib_path):
                    continue

                try:
                    # Use RTLD_GLOBAL so the symbols are available to other libraries
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    loaded_libs.append(os.path.basename(lib_path))
                    if verbose:
                        print(f"DAQ: Preloaded {lib_path}")
                    loaded_this_pattern = True
                    break
                except OSError as e:
                    if verbose:
                        print(f"DAQ: Failed to preload {lib_path}: {e}")
                    continue

    if loaded_libs:
        print(f"DAQ: Preloaded {len(loaded_libs)} CUDA libraries for GPU support")
    else:
        if verbose:
            print("DAQ: No pip CUDA libraries to preload, relying on system CUDA")

    # Always return success - if preloading fails, system CUDA might still work
    _cuda_preload_success = True
    return True


def download_model(dest_path: Path, url: str = MODEL_URL) -> bool:
    """
    Download the ONNX model file.
    
    Parameters
    ----------
    dest_path : Path
        Destination path for the model file
    url : str
        URL to download from
        
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    import urllib.request
    import sys
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading DAQ model from {url}...")
    print(f"Destination: {dest_path}")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest_path), reporthook=report_progress)
        print("\nDownload complete!")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute softmax along specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class DAQOnnxModel:
    """
    ONNX Runtime wrapper for DAQ model inference.
    
    This class handles loading the ONNX model and running batched inference.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file
        
    Attributes
    ----------
    session : onnxruntime.InferenceSession
        The ONNX Runtime inference session
    provider : str
        The execution provider being used (CUDAExecutionProvider or CPUExecutionProvider)
    """
    
    def __init__(self, model_path: str, verbose: bool = False,
                 gpu_id: int = 0, backend: str = "cpu"):
        """Build an ORT inference session for a resolved ORT backend.

        Parameters
        ----------
        backend : str
            One of {"tensorrt", "cuda", "directml", "cpu"}. Must be
            already resolved — "auto" is rejected. Raises RuntimeError
            if the requested EP isn't available on this host.
        gpu_id : int
            CUDA/TRT device id. Ignored for cpu/directml backends.
        """
        if backend not in _ORT_BACKENDS:
            raise ValueError(
                f"DAQOnnxModel: backend must be one of {sorted(_ORT_BACKENDS)},"
                f" got '{backend}'. Use load_model() for 'auto' resolution.")

        # Preload CUDA libraries before importing ORT, but only for GPU
        # backends. CPU-only sessions don't need (and shouldn't trigger)
        # CUDA driver init.
        if backend in ("tensorrt", "cuda"):
            _preload_cuda_libraries(verbose=verbose)

        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.gpu_id = gpu_id
        self.backend = backend

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not verbose:
            sess_options.log_severity_level = 3

        # On macOS the only ORT path that runs is CPU (Mac GPU goes via
        # MLX in mlx_model.py). Tune the CPU pool to all cores.
        if PLATFORM == 'darwin' and backend == 'cpu':
            try:
                n_cpu = os.cpu_count() or 8
            except Exception:
                n_cpu = 8
            sess_options.intra_op_num_threads = max(1, n_cpu)
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        ep = _ep_for_backend(backend)
        available = ort.get_available_providers()
        if backend != "cpu" and ep not in available:
            raise RuntimeError(
                f"Requested backend '{backend}' ({ep}) is not available "
                f"on this host. Installed providers: {available}")

        # Build provider list. GPU EPs include CPU as a safety net for
        # ops that don't have a GPU implementation (ORT routes per-op).
        if backend == "tensorrt":
            providers = [(ep, _trt_provider_options(gpu_id)),
                         "CPUExecutionProvider"]
        elif backend == "cuda":
            providers = [(ep, {"device_id": gpu_id}),
                         "CPUExecutionProvider"]
        elif backend == "directml":
            providers = [ep, "CPUExecutionProvider"]
        else:
            providers = [ep]

        print(f"DAQ: backend='{backend}' -> {ep}")
        self.provider = ep  # tentative; reconciled with actual below

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        # Log the actual provider being used (in case fallback occurred)
        # and update self.provider so downstream code (auto-batch sizing,
        # etc.) sees the EP that's actually running, not the one we asked
        # for. ORT-TRT EP can silently fall back to CUDA EP if TensorRT
        # libs aren't dlopenable at session create.
        actual_provider = self.session.get_providers()[0]
        if actual_provider != self.provider:
            print(f"DAQ: Note: Actual execution provider is {actual_provider} "
                  f"(requested {self.provider})")
            self.provider = actual_provider

        # Log the physical GPU we landed on (if any) and warn on weak combos.
        if self.provider in ("TensorrtExecutionProvider",
                             "CUDAExecutionProvider",
                             "DmlExecutionProvider"):
            name, sm = _query_gpu_info(gpu_id)
            if name is not None:
                sm_str = f"sm_{sm}" if sm is not None else "sm_?"
                print(f"DAQ: GPU: {name} ({sm_str}, device_id={gpu_id})")
                if (self.provider == "CUDAExecutionProvider"
                        and sm is not None and sm < 80):
                    print(f"DAQ: WARNING: CUDA EP on {name} ({sm_str}) is "
                          f"slow for 3D Conv. The cuDNN heuristic picks "
                          f"poor algorithms for tiny 11^3 kernels on "
                          f"Turing-or-older GPUs (compute capability < 8.0). "
                          f"Use the TensorRT backend instead for >5x "
                          f"throughput on this GPU class.")

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]


    def __repr__(self) -> str:
        return f"DAQOnnxModel(model={self.model_path.name}, provider={self.provider})"
    
    def predict(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a batch of patches.
        
        Parameters
        ----------
        patches : np.ndarray
            Input patches with shape (N, 1, 11, 11, 11), float32
            
        Returns
        -------
        tuple of np.ndarray
            (aa_probs, atom_probs, ss_probs) - softmax probabilities
            aa_probs: (N, 20) amino acid probabilities
            atom_probs: (N, 6) atom type probabilities
            ss_probs: (N, 3) secondary structure probabilities
        """
        # Ensure correct dtype and shape
        if patches.dtype != np.float32:
            patches = patches.astype(np.float32)
        
        if patches.ndim == 4:
            # Add channel dimension: (N, D, H, W) -> (N, 1, D, H, W)
            patches = patches[:, np.newaxis, :, :, :]
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: patches})
        
        # Apply softmax to logits
        aa_probs = softmax(outputs[0], axis=1)
        atom_probs = softmax(outputs[1], axis=1)
        ss_probs = softmax(outputs[2], axis=1)
        
        return aa_probs, atom_probs, ss_probs
    
    def get_optimal_batch_size(self, n_patches: int, patch_shape: tuple = (1, 11, 11, 11)) -> int:
        """
        Return the fixed optimal batch size for this EP, bounded by the
        actual workload size. Per-EP defaults come from
        tools/profile_batch_size.py (RTX 6000 Ada, 11^3 patches):

          TensorRT EP    2048   peak ~817K p/s, flat through 4096
          CUDA EP        1024   peak ~308K p/s; runtime "Fallback mode"
                                cliff at 2048+ drops perf to 19K p/s
          DirectML EP    256    conservative (no benchmark data)
          CPU EP         256    latency >100 ms past 256

        Override with DAQ_BATCH_OVERRIDE=<int> for benchmarking.
        """
        import os as _os
        override = _os.environ.get("DAQ_BATCH_OVERRIDE")
        if override:
            try:
                batch_size = max(1, int(override))
                print(f"DAQ: Batch size: {batch_size} (DAQ_BATCH_OVERRIDE)")
                return batch_size
            except ValueError:
                pass

        default_per_provider = {
            "TensorrtExecutionProvider": 2048,
            "CUDAExecutionProvider":     1024,
            "DmlExecutionProvider":      256,
            "CPUExecutionProvider":      256,
        }
        cap = default_per_provider.get(self.provider, 256)
        batch_size = min(cap, max(1, n_patches))
        print(f"DAQ: Batch size: {batch_size} ({self.provider} default {cap})")
        return batch_size

    def predict_batched(
        self,
        patches: np.ndarray,
        batch_size: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batched inference on patches.

        Parameters
        ----------
        patches : np.ndarray
            Input patches with shape (N, 1, 11, 11, 11)
        batch_size : int
            Batch size for inference. If 0 or None, automatically determined.
        progress_callback : callable, optional
            Function called with (current, total) for progress updates

        Returns
        -------
        tuple of np.ndarray
            (aa_probs, atom_probs, ss_probs) concatenated for all patches
        """
        N = patches.shape[0]

        # Auto-determine batch size if not specified
        if batch_size is None or batch_size <= 0:
            batch_size = self.get_optimal_batch_size(N, patches.shape[1:])

        # OOM retry loop: if the chosen batch_size blows GPU memory, halve
        # and retry from scratch. Repeats down to batch=1 before giving up.
        # User-specified batch sizes get the same treatment so a too-large
        # explicit value degrades gracefully instead of crashing.
        original_batch = batch_size
        while batch_size >= 1:
            try:
                return self._predict_loop(patches, batch_size, progress_callback)
            except (RuntimeError, MemoryError) as exc:
                if not _is_oom_error(exc):
                    raise
                if batch_size == 1:
                    raise RuntimeError(
                        f"DAQ: OOM even at batch_size=1; cannot recover. "
                        f"Original error: {exc}") from exc
                new_batch = max(1, batch_size // 2)
                print(f"DAQ: OOM at batch={batch_size}, retrying at "
                      f"batch={new_batch} (started at {original_batch})")
                batch_size = new_batch

    def _predict_loop(self, patches, batch_size, progress_callback):
        """Inner loop that may raise OOM; predict_batched wraps + retries."""
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


def get_model_path(auto_download: bool = True) -> Path:
    """
    Get the path to the ONNX model, checking multiple locations.
    If not found and auto_download is True, downloads to user directory.
    
    Search order:
    1. Environment variable DAQ_MODEL_PATH
    2. Plugin data/ directory (installed package layout)
    3. Plugin data/ directory (development layout)
    4. User's home directory ~/.chimerax/daq_model/Multimodel.onnx
    
    Parameters
    ----------
    auto_download : bool
        If True, automatically download the model if not found
    
    Returns
    -------
    Path
        Path to Multimodel.onnx
    """
    import os
    
    # Possible model locations
    candidates = []
    
    # 1. Environment variable (highest priority)
    env_path = os.environ.get("DAQ_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))
    
    # 2. Installed package layout: data/ is sibling to module
    # When installed: chimerax/daqcolor/onnx_model.py -> chimerax/daqcolor/data/
    module_dir = Path(__file__).parent
    candidates.append(module_dir / "data" / MODEL_FILENAME)
    
    # 3. Development layout: src/ and data/ are siblings under daqcolor/
    # In dev: daqcolor/src/onnx_model.py -> daqcolor/data/
    candidates.append(module_dir.parent / "data" / MODEL_FILENAME)
    
    # 4. User's ChimeraX config directory (also download destination)
    home = Path.home()
    user_model_path = home / ".chimerax" / "daq_model" / MODEL_FILENAME
    candidates.append(user_model_path)
    
    # Return first existing path
    for path in candidates:
        if path.exists():
            return path
    
    # Model not found - try to download if enabled
    if auto_download:
        print("DAQ model not found in any of the expected locations.")
        print("Attempting to download...")
        if download_model(user_model_path):
            return user_model_path
    
    # Return the user path for error message
    return user_model_path


def load_model(model_path: Optional[str] = None, verbose: bool = False,
               backend: str = "auto", gpu_id: int = 0):
    """
    Load the DAQ model with caching. Returns DAQOnnxModel or DAQMLXModel
    depending on the resolved backend — both expose the same surface used
    by compute.py (predict_batched, get_optimal_batch_size, provider).

    Parameters
    ----------
    model_path : str, optional
        Path to ONNX model. Only consulted for ORT backends; MLX uses
        its own weights file (Multimodel.mlx.npz).
    verbose : bool
        Show detailed ONNX/MLX initialization logs.
    backend : str
        Single source of truth for the inference path. Canonical values:
        {"auto", "tensorrt", "cuda", "directml", "mlx", "mlx-cpu", "cpu"}.
        - "auto" walks the platform chain (see _auto_chain).
        - Forced values raise if the backend can't load.
    gpu_id : int
        NVIDIA device id. Used by tensorrt/cuda. Ignored elsewhere.

    Returns
    -------
    DAQOnnxModel | DAQMLXModel
        Loaded model ready for inference. Both have predict_batched().

    Raises
    ------
    FileNotFoundError
        Model weights missing and auto-download failed.
    RuntimeError
        Forced backend unavailable, or auto chain exhausted without
        any backend loading.
    """
    global _model_cache

    backend = (backend or "auto").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. "
                         f"Choose from: {sorted(VALID_BACKENDS)}")

    # Walk auto chain or single forced backend. First success wins.
    chain = _auto_chain() if backend == "auto" else [backend]
    errors = []
    for cand in chain:
        try:
            return _load_or_cache(cand, model_path, gpu_id, verbose)
        except Exception as exc:
            errors.append((cand, exc))
            if backend != "auto":
                # Forced backend: propagate immediately. User asked for this
                # specific path, no silent fallback.
                raise
            print(f"DAQ: backend '{cand}' unavailable ({type(exc).__name__}: "
                  f"{exc}); trying next in chain")

    # All chain entries failed — surface the last error with full context.
    detail = "; ".join(f"{c}: {type(e).__name__}: {e}" for c, e in errors)
    raise RuntimeError(
        f"DAQ: auto backend chain {chain} all failed. Details: {detail}")


def _load_or_cache(backend: str, model_path: Optional[str],
                   gpu_id: int, verbose: bool):
    """Dispatch a single resolved backend; honor + populate the cache.

    Cache key is (backend, gpu_id, file_path_str). file_path differs by
    backend (MLX weights vs ONNX), so identical configs collapse cleanly
    regardless of caller's path argument.
    """
    global _model_cache

    if backend in _MLX_BACKENDS:
        from .mlx_model import DAQMLXModel, get_mlx_weights_path
        mlx_weights = get_mlx_weights_path(auto_download=True)
        if mlx_weights is None:
            raise FileNotFoundError(
                "MLX weights (Multimodel.mlx.npz) not found locally and "
                "could not be downloaded from Hugging Face. Check the network "
                "connection, or set DAQ_MLX_WEIGHTS to a local copy. The file "
                "can also be regenerated with tools/convert_pth_to_mlx.py.")
        file_str = str(mlx_weights.resolve())
        cache_key = (backend, 0, file_str)
        if cache_key in _model_cache:
            print(f"DAQ: Using cached {backend} model")
            return _model_cache[cache_key]
        _evict_other_cache_entries(cache_key)
        mlx_device = "gpu" if backend == "mlx" else "cpu"
        print(f"DAQ: Loading MLX backend (device={mlx_device})...")
        model = DAQMLXModel(file_str, verbose=verbose, device=mlx_device)
        # Forced "mlx" requires Metal. DAQMLXModel silently downgrades on
        # init if Metal probe fails — surface that as an error here so the
        # forced-backend contract holds (user expected GPU, got CPU).
        if backend == "mlx" and getattr(model, "device", "gpu") != "gpu":
            raise RuntimeError(
                "MLX Metal backend requested but Metal unavailable on this "
                "host (only Apple Silicon Macs with a Metal-capable GPU "
                "support MLX Metal). Use backend='mlx-cpu' instead.")
        _model_cache[cache_key] = model
        return model

    # ORT backend path.
    onnx_path: Path
    if model_path is None:
        onnx_path = get_model_path(auto_download=True)
    else:
        onnx_path = Path(model_path)
    if not onnx_path.exists():
        home_path = Path.home() / ".chimerax" / "daq_model" / MODEL_FILENAME
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n\n"
            f"The DAQ score computation requires the ONNX model file.\n"
            f"Automatic download failed. Please install manually:\n"
            f"  1. Download from: {MODEL_URL}\n"
            f"  2. Save to: {home_path}\n"
            f"  3. Or set DAQ_MODEL_PATH environment variable"
        )

    file_str = str(onnx_path.resolve())
    # CPU/DirectML don't use gpu_id; normalize so different gpu_id values
    # don't fragment the cache for backends that ignore it.
    cache_gpu = gpu_id if backend in ("tensorrt", "cuda") else 0
    cache_key = (backend, cache_gpu, file_str)
    if cache_key in _model_cache:
        print(f"DAQ: Using cached model (backend={backend}, "
              f"gpu_id={cache_gpu})")
        return _model_cache[cache_key]
    _evict_other_cache_entries(cache_key)
    print(f"DAQ: Loading model (backend={backend}; first time may take a "
          f"moment for GPU initialization)...")
    model = DAQOnnxModel(file_str, verbose=verbose,
                         gpu_id=gpu_id, backend=backend)
    _model_cache[cache_key] = model
    return model


def _evict_other_cache_entries(keep_key):
    """Drop every cached model except `keep_key`. Frees their VRAM."""
    global _model_cache
    if not _model_cache:
        return
    old_keys = [k for k in list(_model_cache.keys()) if k != keep_key]
    if not old_keys:
        return
    for k in old_keys:
        _release_model(_model_cache.pop(k))
    import gc
    gc.collect()
    print(f"DAQ: Released {len(old_keys)} previously cached model(s) "
          f"to free VRAM")


def _release_model(model):
    """Drop framework-internal references so GC + CUDA driver can free.

    Without this, just removing the dict entry leaves Python references
    from ort.InferenceSession (which holds the CUDA context + workspace)
    or mlx.nn.Module (which holds Metal/CUDA buffers via mx.array).
    """
    # ORT session: deleting the reference is what releases the CUDA
    # context + workspace. Newer ORT exposes _sess.cleanup_session_state()
    # but it's not needed for our purposes.
    for attr in ("session", "model"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass


def clear_model_cache():
    """Drop all cached models and force-free GPU/Metal memory.

    Use after switching backends in the GUI, finishing a heavy job, or
    when another tool needs the GPU. Cost: next inference reloads the
    model (ORT-CUDA ~1 s, ORT-TRT ~0.7 s warm cache / ~11 s cold,
    MLX <1 s).
    """
    global _model_cache
    n = len(_model_cache)
    for _, model in list(_model_cache.items()):
        _release_model(model)
    _model_cache.clear()
    import gc
    gc.collect()
    # Best-effort: tell MLX to release its Metal/CUDA pool.
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    print(f"DAQ: Model cache cleared ({n} entries released, GC ran)")
