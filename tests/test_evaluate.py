"""Check subprocess diagnostics without loading coursework data."""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate import run_stage


class EvaluationProcessTests(unittest.TestCase):
    def test_success_in_requested_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, '-c',
                       'from pathlib import Path; Path("child.txt").write_text("ok")']
            seconds = run_stage(command, os.environ.copy(), temporary, 'Prediction')
            self.assertGreaterEqual(seconds, 0)
            self.assertEqual((Path(temporary) / 'child.txt').read_text(), 'ok')

    def test_failure_preserves_exit_code_and_child_diagnostics(self):
        command = [sys.executable, '-c',
                   'import sys; print("input context"); '
                   'print("missing input file", file=sys.stderr); sys.exit(7)']
        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics), self.assertRaises(SystemExit) as caught:
            run_stage(command, os.environ.copy(), Path.cwd(), 'Prediction')
        self.assertEqual(caught.exception.code, 7)
        for message in ('Prediction failed', 'input context', 'missing input file'):
            self.assertIn(message, diagnostics.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
