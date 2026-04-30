import unittest
from unittest.mock import patch
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tune_gemm_config import main, get_int_input

class TestTuneGemmConfig(unittest.TestCase):
    @patch('sys.argv', ['tune_gemm_config.py', '--custom'])
    @patch('builtins.input', side_effect=['y', 'not_a_number', '2', '32', '4', 'done', 'n'])
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_custom_mode_invalid_input(self, mock_stdout, mock_input):
        main()
        output = mock_stdout.getvalue()
        self.assertIn("❌ Invalid input. Please enter a valid integer.", output)

    @patch('builtins.input', side_effect=['invalid', '42'])
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_get_int_input_invalid_then_valid(self, mock_stdout, mock_input):
        result = get_int_input("Enter value: ")
        self.assertEqual(result, 42)
        self.assertIn("❌ Invalid input. Please enter a valid integer.", mock_stdout.getvalue())

if __name__ == '__main__':
    unittest.main()
