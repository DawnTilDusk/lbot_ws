import unittest
from pathlib import Path
from unittest.mock import patch
from nut_yolo import resolve_python, WORKSPACE
from nut_robot import TaskError


class PythonResolutionTests(unittest.TestCase):
    def test_auto_prefers_workspace_venv(self):
        with patch.object(Path, 'is_file', return_value=True), patch('nut_yolo.os.access', return_value=True):
            self.assertEqual(resolve_python({'python': 'auto'}), WORKSPACE/'.venv-yolo/bin/python')

    def test_auto_uses_conda_when_venv_missing(self):
        conda=Path.home()/'miniconda3/envs/nut-yolo/bin/python'
        with patch.object(Path, 'is_file', lambda p: p==conda), patch('nut_yolo.os.access', return_value=True):
            self.assertEqual(resolve_python({}),conda)

    def test_explicit_path_and_missing_auto(self):
        self.assertEqual(resolve_python({'python':'/custom/python'}),Path('/custom/python'))
        with patch.object(Path, 'is_file', return_value=False):
            with self.assertRaises(TaskError):resolve_python({'python':'auto'})
