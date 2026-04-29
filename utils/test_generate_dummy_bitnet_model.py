import unittest
import sys
import os
import importlib.util

# Dynamically load the module because it has hyphens in its name
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate-dummy-bitnet-model.py')
spec = importlib.util.spec_from_file_location("generate_dummy_bitnet_model", module_path)
generate_dummy_model = importlib.util.module_from_spec(spec)
# Add to sys.modules to allow potential internal imports to resolve correctly if needed
sys.modules["generate_dummy_bitnet_model"] = generate_dummy_model
spec.loader.exec_module(generate_dummy_model)
Model = generate_dummy_model.Model

class TestModelMethods(unittest.TestCase):
    def test_from_model_architecture_unsupported(self):
        """Test that from_model_architecture raises NotImplementedError for invalid architectures."""
        with self.assertRaisesRegex(NotImplementedError, "Architecture 'invalid_arch' not supported!"):
            Model.from_model_architecture('invalid_arch')

if __name__ == '__main__':
    unittest.main()
