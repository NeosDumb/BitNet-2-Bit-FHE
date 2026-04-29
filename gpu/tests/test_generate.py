import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# We need to mock ctypes.CDLL before importing model (which is imported by generate)
with patch('ctypes.CDLL') as mock_cdll:
    mock_lib = MagicMock()
    mock_cdll.return_value = mock_lib

    # Add the parent directory to sys.path so we can import generate
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import generate

class TestGenerate(unittest.TestCase):
    @patch('builtins.input')
    @patch('sys.exit')
    @patch('builtins.print')
    def test_get_prompts_interactive_eof(self, mock_print, mock_exit, mock_input):
        mock_input.side_effect = EOFError
        mock_exit.side_effect = SystemExit  # Raise SystemExit to break out of the generator correctly

        generator = generate.get_prompts(interactive=True)

        with self.assertRaises(SystemExit):
            next(generator)

        mock_print.assert_called_with("exiting")
        mock_exit.assert_called_with(0)

if __name__ == '__main__':
    unittest.main()
