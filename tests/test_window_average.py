import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


class _DummyArg:
    pass


def _dummy_cmd_desc(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}


def _dummy_or(*args, **kwargs):
    return ("Or", args, kwargs)


def _install_chimerax_stubs():
    commands = types.ModuleType("chimerax.core.commands")
    for name in [
        "StringArg",
        "IntArg",
        "BoolArg",
        "ColormapArg",
        "FloatArg",
        "ModelArg",
        "OpenFileNameArg",
        "SaveFileNameArg",
    ]:
        setattr(commands, name, _DummyArg)
    commands.CmdDesc = _dummy_cmd_desc
    commands.Or = _dummy_or
    commands.run = lambda *args, **kwargs: None

    colors = types.ModuleType("chimerax.core.colors")
    colors.Colormap = object
    colors.Color = object

    markers = types.ModuleType("chimerax.markers")
    markers.MarkerSet = object

    map_mod = types.ModuleType("chimerax.map")
    map_mod.MapArg = _DummyArg
    map_mod.Volume = type("Volume", (), {})

    atomic = types.ModuleType("chimerax.atomic")
    atomic.AtomicStructureArg = _DummyArg

    sys.modules.update(
        {
            "chimerax": types.ModuleType("chimerax"),
            "chimerax.core": types.ModuleType("chimerax.core"),
            "chimerax.core.commands": commands,
            "chimerax.core.colors": colors,
            "chimerax.markers": markers,
            "chimerax.map": map_mod,
            "chimerax.atomic": atomic,
        }
    )


def _load_cmd_module():
    _install_chimerax_stubs()

    pkg = types.ModuleType("daqcolor")
    pkg.__path__ = []
    subpkg = types.ModuleType("daqcolor.src")
    subpkg.__path__ = []

    util = types.ModuleType("daqcolor.src.util")
    util._residue_coords = lambda *args, **kwargs: None

    arrow = types.ModuleType("daqcolor.src.arrow")
    arrow.daq_arrowwin = lambda *args, **kwargs: None
    arrow.daq_clearrestraints = lambda *args, **kwargs: None

    sys.modules.update(
        {
            "daqcolor": pkg,
            "daqcolor.src": subpkg,
            "daqcolor.src.util": util,
            "daqcolor.src.arrow": arrow,
        }
    )

    cmd_path = Path(__file__).resolve().parents[1] / "daqcolor" / "src" / "cmd.py"
    spec = importlib.util.spec_from_file_location("daqcolor.src.cmd", cmd_path)
    cmd = importlib.util.module_from_spec(spec)
    sys.modules["daqcolor.src.cmd"] = cmd
    spec.loader.exec_module(cmd)
    return cmd


class _Residue:
    def __init__(self, chain_id, number):
        self.chain_id = chain_id
        self.number = number


def _naive_window_average(residues, scal, half_window):
    out = np.full(len(residues), np.nan, dtype=np.float32)
    scal = np.asarray(scal, dtype=np.float32)
    chain_ids = np.array([r.chain_id for r in residues], dtype=object)
    resnums = np.array([r.number for r in residues], dtype=int)

    for i in range(len(residues)):
        mask = (
            (chain_ids == chain_ids[i])
            & (resnums >= resnums[i] - half_window)
            & (resnums <= resnums[i] + half_window)
        )
        vals = scal[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = vals.mean()

    return out


class WindowAverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmd = _load_cmd_module()

    def assert_matches_naive(self, residues, scal, half_window):
        expected = _naive_window_average(residues, scal, half_window)
        actual = self.cmd._window_average_scal(residues, scal, half_window)
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_sequential_residue_numbers(self):
        residues = [_Residue("A", n) for n in range(1, 8)]
        self.assert_matches_naive(residues, np.arange(1, 8), 2)

    def test_missing_residue_numbers(self):
        residues = [_Residue("A", n) for n in [1, 2, 3, 7, 8, 9]]
        self.assert_matches_naive(residues, [1, 2, 3, 7, 8, 9], 2)

    def test_multiple_chains_do_not_mix(self):
        residues = [
            _Residue("A", 1),
            _Residue("A", 2),
            _Residue("B", 1),
            _Residue("B", 2),
        ]
        self.assert_matches_naive(residues, [1, 2, 10, 20], 1)

    def test_nan_and_inf_are_ignored(self):
        residues = [_Residue("A", n) for n in range(1, 6)]
        self.assert_matches_naive(residues, [np.nan, np.inf, 3, np.nan, 5], 1)

    def test_half_window_zero_uses_same_residue_number(self):
        residues = [
            _Residue("A", 1),
            _Residue("A", 1),
            _Residue("A", 2),
            _Residue("B", 1),
        ]
        self.assert_matches_naive(residues, [1, 3, 10, 100], 0)

    def test_window_plan_cache_keeps_one_entry(self):
        residues_a = [_Residue("A", n) for n in [1, 2, 3]]
        residues_b = [_Residue("B", n) for n in [1, 2, 3]]
        self.cmd._window_average_scal(residues_a, [1, 2, 3], 1)
        self.cmd._window_average_scal(residues_b, [1, 2, 3], 1)
        self.assertEqual(len(self.cmd._WINDOW_AVERAGE_CACHE), 1)


if __name__ == "__main__":
    unittest.main()
