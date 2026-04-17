import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import os

# Import run_inference module
import run_inference

class TestRunInference(unittest.TestCase):

    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        # Test success case for run_command
        run_inference.run_command(['ls', '-l'])
        mock_run.assert_called_once_with(['ls', '-l'], shell=False, check=True)

    @patch('subprocess.run')
    @patch('sys.exit')
    @patch('builtins.print')
    def test_run_command_failure(self, mock_print, mock_exit, mock_run):
        # Test failure case for run_command
        mock_run.side_effect = subprocess.CalledProcessError(1, ['ls', '-l'])

        run_inference.run_command(['ls', '-l'])

        mock_print.assert_called()
        mock_exit.assert_called_once_with(1)

    @patch('platform.system')
    @patch('run_inference.args', create=True)
    @patch('run_inference.run_command')
    def test_run_inference_linux(self, mock_run_command, mock_args, mock_system):
        # Test run_inference on Linux
        mock_system.return_value = 'Linux'
        mock_args.model = 'test_model.gguf'
        mock_args.n_predict = 128
        mock_args.threads = 4
        mock_args.prompt = 'Hello world'
        mock_args.ctx_size = 512
        mock_args.temperature = 0.7
        mock_args.conversation = False

        run_inference.run_inference()

        expected_command = [
            os.path.join('build', 'bin', 'llama-cli'),
            '-m', 'test_model.gguf',
            '-n', '128',
            '-t', '4',
            '-p', 'Hello world',
            '-ngl', '0',
            '-c', '512',
            '--temp', '0.7',
            "-b", "1",
        ]
        mock_run_command.assert_called_once_with(expected_command)

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('run_inference.args', create=True)
    @patch('run_inference.run_command')
    def test_run_inference_windows_exists(self, mock_run_command, mock_args, mock_exists, mock_system):
        # Test run_inference on Windows when the preferred path exists
        mock_system.return_value = 'Windows'
        mock_exists.return_value = True
        mock_args.model = 'test_model.gguf'
        mock_args.n_predict = 128
        mock_args.threads = 2
        mock_args.prompt = 'Windows test'
        mock_args.ctx_size = 2048
        mock_args.temperature = 0.8
        mock_args.conversation = False

        run_inference.run_inference()

        expected_path = os.path.join('build', 'bin', 'Release', 'llama-cli.exe')
        self.assertEqual(mock_run_command.call_args[0][0][0], expected_path)

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('run_inference.args', create=True)
    @patch('run_inference.run_command')
    def test_run_inference_windows_not_exists(self, mock_run_command, mock_args, mock_exists, mock_system):
        # Test run_inference on Windows when the preferred path does NOT exist
        mock_system.return_value = 'Windows'
        mock_exists.return_value = False
        mock_args.model = 'test_model.gguf'
        mock_args.n_predict = 128
        mock_args.threads = 2
        mock_args.prompt = 'Windows test'
        mock_args.ctx_size = 2048
        mock_args.temperature = 0.8
        mock_args.conversation = False

        run_inference.run_inference()

        expected_path = os.path.join('build', 'bin', 'llama-cli')
        self.assertEqual(mock_run_command.call_args[0][0][0], expected_path)

    @patch('platform.system')
    @patch('run_inference.args', create=True)
    @patch('run_inference.run_command')
    def test_run_inference_conversation(self, mock_run_command, mock_args, mock_system):
        # Test run_inference with conversation flag
        mock_system.return_value = 'Linux'
        mock_args.model = 'test_model.gguf'
        mock_args.n_predict = 128
        mock_args.threads = 2
        mock_args.prompt = 'Chat test'
        mock_args.ctx_size = 2048
        mock_args.temperature = 0.8
        mock_args.conversation = True

        run_inference.run_inference()

        self.assertIn('-cnv', mock_run_command.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
