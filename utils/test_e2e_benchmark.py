import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import subprocess
import argparse

# Add the directory containing e2e_benchmark to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import e2e_benchmark

class TestE2EBenchmark(unittest.TestCase):
    def setUp(self):
        # Reset the args in e2e_benchmark to avoid cross-test interference
        e2e_benchmark.args = None

    def test_parse_args(self):
        # Test with required argument -m
        test_args = ['-m', 'test_model.gguf']
        with patch('sys.argv', ['e2e_benchmark.py'] + test_args):
            args = e2e_benchmark.parse_args()
            self.assertEqual(args.model, 'test_model.gguf')
            self.assertEqual(args.n_token, 128)
            self.assertEqual(args.n_prompt, 512)
            self.assertEqual(args.threads, 2)

        # Test with all arguments
        test_args = ['-m', 'test_model.gguf', '-n', '256', '-p', '1024', '-t', '4']
        with patch('sys.argv', ['e2e_benchmark.py'] + test_args):
            args = e2e_benchmark.parse_args()
            self.assertEqual(args.model, 'test_model.gguf')
            self.assertEqual(args.n_token, 256)
            self.assertEqual(args.n_prompt, 1024)
            self.assertEqual(args.threads, 4)

        # Test with missing required argument
        test_args = []
        with patch('sys.argv', ['e2e_benchmark.py'] + test_args):
            with patch('sys.stderr', new=MagicMock()):  # Suppress error message
                with self.assertRaises(SystemExit):
                    e2e_benchmark.parse_args()

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    def test_run_command(self, mock_join, mock_file_open, mock_run):
        # Setup mock args for e2e_benchmark
        e2e_benchmark.args = MagicMock()
        e2e_benchmark.args.log_dir = 'logs'

        # Scenario 1: run_command without log_step (success)
        e2e_benchmark.run_command(['ls'])
        mock_run.assert_called_with(['ls'], shell=False, check=True)

        # Scenario 2: run_command without log_step (failure)
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ls')
        with patch('sys.exit') as mock_exit:
            e2e_benchmark.run_command(['ls'])
            mock_exit.assert_called_with(1)

        # Scenario 3: run_command with log_step (success)
        mock_run.side_effect = None
        mock_join.return_value = 'logs/step.log'
        e2e_benchmark.run_command(['ls'], log_step='step')
        mock_file_open.assert_called_with('logs/step.log', 'w')
        mock_run.assert_called()

        # Scenario 4: run_command with log_step (failure)
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ls')
        with patch('sys.exit') as mock_exit:
            e2e_benchmark.run_command(['ls'], log_step='step')
            mock_exit.assert_called_with(1)

    @patch('e2e_benchmark.run_command')
    @patch('os.path.exists')
    @patch('platform.system')
    def test_run_benchmark(self, mock_system, mock_exists, mock_run_command):
        # Setup mock args
        e2e_benchmark.args = MagicMock()
        e2e_benchmark.args.model = 'model.gguf'
        e2e_benchmark.args.n_token = 128
        e2e_benchmark.args.threads = 2
        e2e_benchmark.args.n_prompt = 512

        # Scenario 1: linux, binary exists
        mock_system.return_value = 'Linux'
        mock_exists.return_value = True
        e2e_benchmark.run_benchmark()
        mock_run_command.assert_called()
        cmd = mock_run_command.call_args[0][0]
        self.assertIn('llama-bench', cmd[0])
        self.assertIn('model.gguf', cmd)

        # Scenario 2: linux, binary doesn't exist
        mock_exists.return_value = False
        with patch('sys.exit') as mock_exit:
            e2e_benchmark.run_benchmark()
            mock_exit.assert_called_with(1)

if __name__ == '__main__':
    unittest.main()
