import unittest
from unittest.mock import patch, MagicMock
import sys
import os

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

if __name__ == '__main__':
    unittest.main()
