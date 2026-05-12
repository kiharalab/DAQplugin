# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
Platform-specific constants and configuration for DAQplugin.

This module provides cross-platform GPU support:
- Linux: NVIDIA CUDA (onnxruntime-gpu)
- Windows: DirectML (onnxruntime-directml) - works with AMD/Intel/NVIDIA
- macOS: MLX/Metal (Apple Silicon) via mlx_model.py; CPU EP fallback
"""

import sys
import subprocess

# =============================================================================
# Platform Detection
# =============================================================================

def get_platform():
    """
    Get the current platform identifier.

    Returns
    -------
    str
        One of: 'linux', 'darwin' (macOS), 'windows'
    """
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform == 'darwin':
        return 'darwin'
    elif sys.platform.startswith('win'):
        return 'windows'
    return 'unknown'


PLATFORM = get_platform()


# =============================================================================
# ONNX Runtime EP availability
# =============================================================================
#
# Backend-name -> EP resolution lives in onnx_model.py (single source of
# truth). This module only exposes raw EP enumeration plus the platform
# GPU detection helpers used by the GUI.

def get_available_providers():
    """List ONNX Runtime execution providers installed in this interpreter."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        return ['CPUExecutionProvider']


# =============================================================================
# GPU Detection
# =============================================================================

def detect_nvidia_gpus():
    """
    Detect available NVIDIA GPUs using nvidia-smi.

    Returns
    -------
    list of dict
        Each dict contains: 'id', 'name', 'memory_total', 'memory_free', 'display_text'
        Returns empty list if no GPUs found or nvidia-smi not available.
    """
    gpus = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        name = parts[1]
                        mem_total = int(parts[2])  # MB
                        mem_free = int(parts[3])   # MB
                        display_text = f"GPU {gpu_id}: {name} ({mem_total//1024}GB, {mem_free//1024}GB free)"
                        gpus.append({
                            'id': gpu_id,
                            'name': name,
                            'memory_total': mem_total,
                            'memory_free': mem_free,
                            'display_text': display_text
                        })
    except Exception:
        pass

    return gpus


def is_gpu_available():
    """
    Check if any GPU acceleration is available.

    Returns
    -------
    bool
        True if any GPU provider is available
    """
    available = get_available_providers()
    gpu_providers = [
        'TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'DmlExecutionProvider',
        'MIGraphXExecutionProvider',
    ]
    # On macOS, MLX/Metal provides GPU via mlx_model.py (not ORT). Require
    # both the package import AND a successful Metal device probe -- ARM
    # Macs in headless/CI/VM contexts may have mlx but no Metal device,
    # and a bare `import mlx.core` does not catch that.
    if PLATFORM == 'darwin':
        try:
            import mlx.core as mx
            with mx.stream(mx.gpu):
                _probe = mx.array([0.0], dtype=mx.float32)
                mx.eval(_probe)
            return True
        except Exception:
            pass
    return any(p in available for p in gpu_providers)


# =============================================================================
# Default Settings
# =============================================================================

DEFAULT_BATCH_SIZE = 0  # Auto-detect
DEFAULT_MAX_POINTS = 500000
DEFAULT_STRIDE = 2
DEFAULT_K = 1
DEFAULT_HALF_WINDOW = 9

DEFAULT_CLAMP_MIN = -1.0
DEFAULT_CLAMP_MAX = 1.0

MODEL_URL = "https://huggingface.co/zhtronics/DAQscore/resolve/main/Multimodel.onnx"
MODEL_FILENAME = "Multimodel.onnx"
