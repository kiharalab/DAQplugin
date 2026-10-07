import ast, types
from pathlib import Path
import numpy as np
import time
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(source):
    tree = ast.parse(source)
    names = {'extract_threshold_points', 'compute_log_ratio_scores', 'compute_daq_scores'}
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    from typing import Optional, Tuple, Union, Callable
    env = dict(np=np, Path=Path, Optional=Optional, Tuple=Tuple, Union=Union, Callable=Callable, time=time, perf_counter=time.perf_counter)
    exec(compile(selected, '<pipeline>', 'exec'), env)
    return env

new = load((ROOT / "daqcolor/src/compute.py").read_text())
class Volume: pass
data = np.linspace(-2, 6, 120, dtype=np.float32).reshape(4, 5, 6)
vol = Volume()
vol.data = types.SimpleNamespace(matrix=lambda: data, origin=(1,2,3), step=(1,1,1))
log = types.SimpleNamespace(info=lambda *a: None, status=lambda *a: None, warning=lambda *a: None)
session = types.SimpleNamespace(logger=log, models=types.SimpleNamespace(list=lambda: [vol]))
def predict(patches, **kw):
    return tuple((patches[:, :1] + np.arange(1, n+1)[None,:]) / (n + 10) for n in (20,6,3))
for env in (new,):
    env.update(resize_map_to_1a=lambda *a: vol, normalize_volume=lambda d:d,
               extract_patches=lambda d,p,*a,**kw: p,
               load_model=lambda *a,**kw:types.SimpleNamespace(predict_batched=predict))

from unittest.mock import patch

class FullVolumeTests(unittest.TestCase):
    def run_pipeline(self, **kwargs):
        with patch.dict(sys.modules, {'chimerax.map': types.SimpleNamespace(Volume=Volume)}):
            return new['compute_daq_scores'](session, vol, **kwargs)

    def test_off_preserves_legacy_sampling_and_scores(self):
        for contour in (-1., 0., 2., 20.):
            for stride in (1, 2):
                for cap in (5, 500):
                    with self.subTest(contour=contour, stride=stride, cap=cap):
                        np.random.seed(23)
                        points, density = new['extract_threshold_points'](data, (1,2,3), (1,1,1), contour * .5, stride, cap)
                        expected = (new['compute_log_ratio_scores'](points, *predict(points), density, contour)
                                    if len(points) else np.zeros((0, 32), dtype=np.float32))
                        for options in ({}, {'full_volume': False}):
                            np.random.seed(23)
                            result = self.run_pipeline(contour=contour, stride=stride, max_points=cap, **options)
                            np.testing.assert_array_equal(result[0], points)
                            np.testing.assert_array_equal(result[1], expected)

    def test_on_covers_volume_and_filters_reference(self):
        for stride in (1, 2):
            points, scores, *_ = self.run_pipeline(contour=2, stride=stride, max_points=500, full_volume=True)
            density = data[::stride, ::stride, ::stride].ravel()
            self.assertEqual(len(points), len(density))
            expected = np.concatenate([points] + [np.log(p / p[density >= 2].mean(axis=0)).astype(np.float32) for p in predict(points)], axis=1)
            np.testing.assert_array_equal(scores, expected)

    def test_on_requires_contour_reference_points(self):
        with self.assertRaisesRegex(ValueError, 'reference probabilities'):
            self.run_pipeline(contour=20, full_volume=True)

    def test_max_points_warning_only_when_full_volume_exceeds_limit(self):
        cases = [
            (True, 1, 119, True),
            (True, 1, 120, False),
            (True, 2, 17, True),
            (True, 2, 18, False),
            (True, 2, None, False),
            (False, 1, 5, False),
        ]
        for full_volume, stride, cap, warned in cases:
            with self.subTest(full_volume=full_volume, stride=stride, cap=cap):
                with patch.object(log, 'warning') as warning:
                    self.run_pipeline(contour=-3, stride=stride, max_points=cap, full_volume=full_volume)
                    self.assertEqual(warning.call_count, int(warned))
                    if warned:
                        message = warning.call_args.args[0]
                        self.assertIn('exceeding Max Points', message)
                        self.assertIn(f'Only {cap:,} randomly selected points', message)
