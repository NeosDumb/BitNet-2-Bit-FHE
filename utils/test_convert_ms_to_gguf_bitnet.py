import unittest
import io
import os
import sys
import importlib.util
from unittest.mock import MagicMock

# Mock missing dependencies to allow loading the module
mock_modules = [
    "numpy",
    "sentencepiece",
    "gguf",
    "convert_utils",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "transformers"
]

for mod_name in mock_modules:
    sys.modules[mod_name] = MagicMock()

# Dynamically load the module because it has hyphens in its name
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convert-ms-to-gguf-bitnet.py')
spec = importlib.util.spec_from_file_location("convert_ms_to_gguf_bitnet", module_path)
convert_ms = importlib.util.module_from_spec(spec)
sys.modules["convert_ms_to_gguf_bitnet"] = convert_ms
spec.loader.exec_module(convert_ms)

class TestMustRead(unittest.TestCase):
    def test_must_read_success(self):
        data = b"hello world"
        fp = io.BytesIO(data)
        result = convert_ms.must_read(fp, 5)
        self.assertEqual(result, b"hello")
        self.assertEqual(fp.tell(), 5)

    def test_must_read_failure(self):
        data = b"tiny"
        fp = io.BytesIO(data)
        with self.assertRaisesRegex(EOFError, "unexpectedly reached end of file"):
            convert_ms.must_read(fp, 10)

if __name__ == '__main__':
    unittest.main()
