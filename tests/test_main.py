"""To run all tests: python -m unittest discover
To run only this test: python -m unittest tests.test_main"""

import subprocess
import unittest


class TestMainScript(unittest.TestCase):

    def test_main_runs(self):
        """Smoke test: ensure main.py runs without crashing."""
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Accuracy", result.stdout)


if __name__ == '__main__':
    unittest.main()
