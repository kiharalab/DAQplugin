"""Backend / execution-provider fallback behavior for onnx_model.

Regression coverage for the "RTX 4080 silently ran on CPU" bug:

ORT's InferenceSession does NOT raise when a GPU EP (TensorRT/CUDA) can't
initialize — it logs "EP Error ... Falling back" and creates a CPU-only
session. That silently defeated both:

  * the forced-backend contract (user picks TensorRT, gets CPU at ~100x
    slowdown, no error), and
  * backend='auto' (the tensorrt candidate "succeeds" as CPU, so the chain
    never advances to the CUDA EP).

These tests pin the fixed behavior: a forced GPU backend whose EP can't load
must RAISE (with an actionable hint), while 'auto' must still resolve to a
working session and 'cpu' must never trip the guard.

They are integration tests: they need onnxruntime + the real DAQ ONNX model
on disk, and they assume the *current host's* GPU EPs cannot initialize
(coyote: driver 470 / CUDA 11.4, ORT built for CUDA 12 -> every GPU EP falls
back to CPU). On a host where the GPU EP loads cleanly the forced-raise tests
self-skip.
"""

import importlib
import sys
import types
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "daqcolor" / "src"
MODEL = Path.home() / ".chimerax" / "daq_model" / "Multimodel.onnx"


def _load_onnx_module():
    """Import daqcolor/src/onnx_model.py with its real .constants sibling."""
    pkg = types.ModuleType("daqcolor")
    pkg.__path__ = []
    subpkg = types.ModuleType("daqcolor.src")
    subpkg.__path__ = [str(SRC)]  # let `.constants` resolve to the real file
    sys.modules.setdefault("daqcolor", pkg)
    sys.modules["daqcolor.src"] = subpkg
    return importlib.import_module("daqcolor.src.onnx_model")


def _ort_available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _gpu_ep_in_build():
    """True if this ORT build registers a GPU EP (i.e. onnxruntime-gpu)."""
    try:
        import onnxruntime as ort
        eps = set(ort.get_available_providers())
        return bool(eps & {"TensorrtExecutionProvider",
                           "CUDAExecutionProvider", "DmlExecutionProvider"})
    except Exception:
        return False


def _gpu_ep_actually_loads(om, backend):
    """True if `backend`'s EP loads a non-CPU session on this host.

    Used to skip the forced-raise tests on hosts where the GPU stack is
    healthy (there the fallback path simply never triggers).
    """
    try:
        m = om.DAQOnnxModel(str(MODEL), backend=backend)
        return m.provider != "CPUExecutionProvider"
    except Exception:
        return False


@unittest.skipUnless(MODEL.exists(), f"DAQ ONNX model not found at {MODEL}")
@unittest.skipUnless(_ort_available(), "onnxruntime not installed")
class BackendFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.om = _load_onnx_module()

    def setUp(self):
        # Each test starts from a clean cache so cross-test cache reuse
        # can't mask a regression.
        self.om.clear_model_cache()

    def test_forced_tensorrt_raises_on_silent_cpu_fallback(self):
        if not _gpu_ep_in_build():
            self.skipTest("onnxruntime-gpu (TensorRT EP) not in this build")
        if _gpu_ep_actually_loads(self.om, "tensorrt"):
            self.skipTest("TensorRT EP loads on this host; no fallback to test")
        self.om.clear_model_cache()
        with self.assertRaises(RuntimeError) as ctx:
            self.om.DAQOnnxModel(str(MODEL), backend="tensorrt")
        msg = str(ctx.exception)
        # Must name the EP and offer an actionable remediation hint.
        self.assertIn("TensorrtExecutionProvider", msg)
        self.assertIn("libnvinfer", msg)

    def test_forced_cuda_raises_on_silent_cpu_fallback(self):
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                self.skipTest("CUDA EP not in this build")
        except Exception:
            self.skipTest("onnxruntime not importable")
        if _gpu_ep_actually_loads(self.om, "cuda"):
            self.skipTest("CUDA EP loads on this host; no fallback to test")
        self.om.clear_model_cache()
        with self.assertRaises(RuntimeError) as ctx:
            self.om.DAQOnnxModel(str(MODEL), backend="cuda")
        self.assertIn("CUDAExecutionProvider", str(ctx.exception))

    def test_forced_cpu_never_raises(self):
        m = self.om.DAQOnnxModel(str(MODEL), backend="cpu")
        self.assertEqual(m.provider, "CPUExecutionProvider")

    def test_auto_resolves_to_working_session(self):
        # auto must walk the chain and return a usable session (CPU here,
        # since no GPU EP loads on this host) rather than crashing.
        m = self.om.load_model(str(MODEL), backend="auto")
        self.assertIn("ExecutionProvider", m.provider)


if __name__ == "__main__":
    unittest.main()
