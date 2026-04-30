import unittest
from unittest.mock import patch
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tune_gemm_config import main

class TestTuneGemmConfig(unittest.TestCase):
    @patch('sys.argv', ['tune_gemm_config.py', '--custom'])
    @patch('builtins.input', side_effect=['y', 'not_a_number', '128', '128', '1', 'done', 'n'])
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_custom_mode_invalid_input(self, mock_stdout, mock_input):
        main()
        output = mock_stdout.getvalue()
        self.assertIn("❌ Invalid input. Please enter a valid integer.", output)

if __name__ == '__main__':
    unittest.main()
