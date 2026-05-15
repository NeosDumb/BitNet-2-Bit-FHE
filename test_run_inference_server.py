import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import os

# Import run_inference_server module
import run_inference_server

class TestRunInferenceServer(unittest.TestCase):

    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        # Test success case for run_command
        run_inference_server.run_command(['ls', '-l'])
        mock_run.assert_called_once_with(['ls', '-l'], shell=False, check=True)


    @patch('subprocess.run')
    @patch('sys.exit')
    @patch('builtins.print')
    def test_run_command_failure(self, mock_print, mock_exit, mock_run):
        # Test failure case for run_command
        mock_run.side_effect = subprocess.CalledProcessError(1, ['ls', '-l'])

        run_inference_server.run_command(['ls', '-l'])

        mock_print.assert_called()
        mock_exit.assert_called_once_with(1)

    @patch('platform.system')
    @patch('run_inference_server.args', create=True)
    @patch('run_inference_server.run_command')
    def test_run_server_linux(self, mock_run_command, mock_args, mock_system):
        # Test run_server on Linux
        mock_system.return_value = 'Linux'
        mock_args.model = 'test_model.gguf'
        mock_args.ctx_size = 2048
        mock_args.threads = 4
        mock_args.n_predict = 128
        mock_args.temperature = 0.7
        mock_args.host = '127.0.0.1'
        mock_args.port = 8080
        mock_args.prompt = None

        run_inference_server.run_server()

        expected_command = [
            os.path.join('build', 'bin', 'llama-server'),
            '-m', 'test_model.gguf',
            '-c', '2048',
            '-t', '4',
            '-n', '128',
            '-ngl', '0',
            '--temp', '0.7',
            '--host', '127.0.0.1',
            '--port', '8080',
            '-cb'
        ]
        mock_run_command.assert_called_once_with(expected_command)

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('run_inference_server.args', create=True)
    @patch('run_inference_server.run_command')
    def test_run_server_windows_exists(self, mock_run_command, mock_args, mock_exists, mock_system):
        # Test run_server on Windows when the preferred path exists
        mock_system.return_value = 'Windows'
        mock_exists.return_value = True
        mock_args.model = 'test_model.gguf'
        mock_args.ctx_size = 2048
        mock_args.threads = 2
        mock_args.n_predict = 4096
        mock_args.temperature = 0.8
        mock_args.host = '0.0.0.0'
        mock_args.port = 9000
        mock_args.prompt = 'System prompt'

        run_inference_server.run_server()

        expected_path = os.path.join('build', 'bin', 'Release', 'llama-server.exe')
        self.assertEqual(mock_run_command.call_args[0][0][0], expected_path)
        self.assertIn('-p', mock_run_command.call_args[0][0])
        self.assertIn('System prompt', mock_run_command.call_args[0][0])

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('run_inference_server.args', create=True)
    @patch('run_inference_server.run_command')
    def test_run_server_windows_not_exists(self, mock_run_command, mock_args, mock_exists, mock_system):
        # Test run_server on Windows when the preferred path does NOT exist
        mock_system.return_value = 'Windows'
        mock_exists.return_value = False
        mock_args.model = 'test_model.gguf'
        mock_args.ctx_size = 2048
        mock_args.threads = 2
        mock_args.n_predict = 4096
        mock_args.temperature = 0.8
        mock_args.host = '127.0.0.1'
        mock_args.port = 8080
        mock_args.prompt = None

        run_inference_server.run_server()

        expected_path = os.path.join('build', 'bin', 'llama-server')
        self.assertEqual(mock_run_command.call_args[0][0][0], expected_path)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_signal_handler(self, mock_print, mock_exit):
        # Test signal_handler
        run_inference_server.signal_handler(None, None)
        mock_print.assert_called_with("Ctrl+C pressed, shutting down server...")
        mock_exit.assert_called_once_with(0)

if __name__ == '__main__':
    unittest.main()
