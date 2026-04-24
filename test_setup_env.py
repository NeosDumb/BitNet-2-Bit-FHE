import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import subprocess

# Import setup_env
import setup_env

class TestSetupEnv(unittest.TestCase):
    @patch('platform.machine')
    def test_parse_args_defaults_x86(self, mock_machine):
        mock_machine.return_value = 'x86_64'
        test_args = ['setup_env.py']
        with patch('sys.argv', test_args):
            args = setup_env.parse_args()
            self.assertEqual(args.model_dir, 'models')
            self.assertEqual(args.log_dir, 'logs')
            self.assertEqual(args.quant_type, 'i2_s')
            self.assertFalse(args.quant_embd)
            self.assertFalse(args.use_pretuned)
            self.assertIsNone(args.hf_repo)

    @patch('platform.machine')
    def test_parse_args_custom_arm64(self, mock_machine):
        mock_machine.return_value = 'arm64'
        test_args = [
            'setup_env.py',
            '--hf-repo', '1bitLLM/bitnet_b1_58-large',
            '--model-dir', 'custom_models',
            '--log-dir', 'custom_logs',
            '--quant-type', 'tl1',
            '--quant-embd',
            '--use-pretuned'
        ]
        with patch('sys.argv', test_args):
            args = setup_env.parse_args()
            self.assertEqual(args.hf_repo, '1bitLLM/bitnet_b1_58-large')
            self.assertEqual(args.model_dir, 'custom_models')
            self.assertEqual(args.log_dir, 'custom_logs')
            self.assertEqual(args.quant_type, 'tl1')
            self.assertTrue(args.quant_embd)
            self.assertTrue(args.use_pretuned)

    @patch('platform.machine')
    def test_parse_args_invalid_hf_repo(self, mock_machine):
        mock_machine.return_value = 'x86_64'
        test_args = ['setup_env.py', '--hf-repo', 'invalid/repo']
        with patch('sys.argv', test_args):
            with patch('sys.stderr', new=MagicMock()):
                with self.assertRaises(SystemExit):
                    setup_env.parse_args()

    @patch('platform.machine')
    def test_parse_args_invalid_quant_type_for_arch(self, mock_machine):
        # tl1 is only for arm64 in setup_env.py
        mock_machine.return_value = 'x86_64'
        test_args = ['setup_env.py', '--quant-type', 'tl1']
        with patch('sys.argv', test_args):
            with patch('sys.stderr', new=MagicMock()):
                with self.assertRaises(SystemExit):
                    setup_env.parse_args()

class TestSetupEnvUtils(unittest.TestCase):
    def setUp(self):
        # setup_env.args is used globally in run_command and get_model_name
        setup_env.args = MagicMock()
        setup_env.args.log_dir = "test_logs"

    @patch('subprocess.run')
    def test_run_command_no_log_step_success(self, mock_run):
        setup_env.run_command(["ls"])
        mock_run.assert_called_once_with(["ls"], shell=False, check=True)

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_command_with_log_step_success(self, mock_file, mock_run):
        setup_env.run_command(["ls"], log_step="test_step")

        expected_log_path = os.path.join("test_logs", "test_step.log")
        mock_file.assert_called_once_with(expected_log_path, "w")
        mock_run.assert_called_once_with(
            ["ls"],
            shell=False,
            check=True,
            stdout=mock_file(),
            stderr=mock_file()
        )

    @patch('subprocess.run')
    def test_run_command_no_log_step_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ls"])
        with patch('logging.error') as mock_log:
            with self.assertRaises(SystemExit) as cm:
                setup_env.run_command(["ls"])
            self.assertEqual(cm.exception.code, 1)
            mock_log.assert_called()

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_command_with_log_step_failure(self, mock_file, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ls"])
        with patch('logging.error') as mock_log:
            with self.assertRaises(SystemExit) as cm:
                setup_env.run_command(["ls"], log_step="test_step")
            self.assertEqual(cm.exception.code, 1)
            mock_log.assert_called()
            expected_log_path = os.path.join("test_logs", "test_step.log")
            mock_file.assert_called_once_with(expected_log_path, "w")

    def test_get_model_name_hf_repo(self):
        setup_env.args.hf_repo = "1bitLLM/bitnet_b1_58-large"
        model_name = setup_env.get_model_name()
        self.assertEqual(model_name, "bitnet_b1_58-large")

    def test_get_model_name_dir(self):
        setup_env.args.hf_repo = None
        setup_env.args.model_dir = "/path/to/my_model/"
        model_name = setup_env.get_model_name()
        self.assertEqual(model_name, "my_model")

if __name__ == '__main__':
    unittest.main()
