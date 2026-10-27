"""Exercise synchronous feedback lifecycle without requiring ChimeraX/Qt."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Widget:
    def __init__(self):
        self.label = 'Calculate'
        self.enabled = True
        self.value = 0

    def text(self): return self.label
    def setText(self, text): self.label = text
    def isEnabled(self): return self.enabled
    def setEnabled(self, enabled): self.enabled = enabled
    def setVisible(self, visible): self.visible = visible
    def setValue(self, value): self.value = value
    def repaint(self): pass


class FeedbackTests(unittest.TestCase):
    def make_tool(self, run):
        path = Path(__file__).resolve().parents[1] / 'daqcolor/src/tool.py'
        tree = ast.parse(path.read_text())
        tool_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DAQTool')
        methods = [n for n in tool_class.body if isinstance(n, ast.FunctionDef) and n.name in
                   ('_update_compute_progress', '_run_compute_with_feedback')]
        self.event_pumps = []
        namespace = {
            'run': run,
            'QCoreApplication': SimpleNamespace(processEvents=self.event_pumps.append),
            'QEventLoop': SimpleNamespace(ExcludeUserInputEvents=1, ExcludeSocketNotifiers=2),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        tool = SimpleNamespace(_compute_running=False, session=SimpleNamespace(),
                               grid_compute_button=Widget(), pdb_compute_button=Widget(),
                               compute_progress=Widget(), compute_status=Widget())
        for method in methods:
            setattr(tool, method.name, namespace[method.name].__get__(tool))
        return tool

    def test_success_and_reentry(self):
        def run(session, command):
            self.assertFalse(tool.grid_compute_button.enabled)
            self.assertFalse(tool.pdb_compute_button.enabled)
            self.assertFalse(tool._run_compute_with_feedback(command, tool.pdb_compute_button))
            session._daq_progress_callback(4.5, 6, 'CNN inference')
            self.assertEqual(tool.compute_progress.value, 74)
            self.assertIn('74%', tool.grid_compute_button.text())
            self.assertEqual(self.event_pumps, [3, 3])
            session._daq_progress_callback(6, 6, 'Done!')
            self.assertEqual(tool.compute_progress.value, 99)
        tool = self.make_tool(run)
        self.assertTrue(tool._run_compute_with_feedback('compute', tool.grid_compute_button))
        self.assertEqual(tool.compute_progress.value, 100)
        self.assertTrue(tool.grid_compute_button.enabled)
        self.assertFalse(tool._compute_running)
        self.assertFalse(hasattr(tool.session, '_daq_progress_callback'))

    def test_failure_restores_state_and_callback(self):
        def run(session, command):
            raise RuntimeError('inference failed')
        tool = self.make_tool(run)
        previous = object()
        tool.session._daq_progress_callback = previous
        tool.pdb_compute_button.enabled = False
        self.assertFalse(tool._run_compute_with_feedback('compute', tool.grid_compute_button))
        self.assertIs(tool.session._daq_progress_callback, previous)
        self.assertTrue(tool.grid_compute_button.enabled)
        self.assertFalse(tool.pdb_compute_button.enabled)
        self.assertEqual(tool.grid_compute_button.text(), 'Calculate')
        self.assertIn('failed', tool.compute_status.text())
        self.assertFalse(tool._compute_running)


if __name__ == '__main__':
    unittest.main()
