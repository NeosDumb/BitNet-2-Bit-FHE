import unittest
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

# --- Robust NumPy Mocking for Restricted Environments ---
# This environment lacks 'numpy', so we provide a minimal implementation
# of the NDArray interface to verify the logic of the 'permute' function.

class FakeNDArray:
    """Simulates a NumPy array with basic reshape and swapaxes support."""
    def __init__(self, data, shape, dtype=None):
        self.data = list(data)
        self.shape = shape
        self.dtype = dtype or MagicMock()
        self.size = len(self.data)

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)

        # In this specific test context, we don't need to handle '-1'
        # as the 'permute' function provides explicit dimensions.
        return FakeNDArray(self.data, new_shape, self.dtype)

    def swapaxes(self, axis1, axis2):
        """Simulates NumPy's swapaxes by re-mapping data indices."""
        # Calculate strides to map between flat data and N-D coordinates
        strides = []
        s = 1
        for dim in reversed(self.shape):
            strides.append(s)
            s *= dim
        strides.reverse()

        # Utility to generate all N-D indices for a given shape
        def get_all_indices(dims):
            if not dims:
                yield []
                return
            for i in range(dims[0]):
                for rest in get_all_indices(dims[1:]):
                    yield [i] + rest

        # Map original coordinates to flat data values
        indexed_data = {}
        for idx in get_all_indices(self.shape):
            flat_idx = sum(a * b for a, b in zip(idx, strides))
            indexed_data[tuple(idx)] = self.data[flat_idx]

        # Determine new shape after swapping axes
        new_shape = list(self.shape)
        new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]

        # Populate new flat data list by iterating through new shape's coordinates
        new_data = []
        for idx in get_all_indices(new_shape):
            # Look up value from original coordinates (with swapped axes)
            original_idx = list(idx)
            original_idx[axis1], original_idx[axis2] = original_idx[axis2], original_idx[axis1]
            new_data.append(indexed_data[tuple(original_idx)])

        return FakeNDArray(new_data, tuple(new_shape), self.dtype)

    def __repr__(self):
        return f"FakeNDArray(data={self.data}, shape={self.shape})"

# Setup the mock module
mock_np = MagicMock()
mock_np.float32 = MagicMock()
mock_np.float32._mock_name = 'float32'
mock_np.float16 = MagicMock()
mock_np.float16._mock_name = 'float16'
mock_np.uint16 = MagicMock()
mock_np.uint16._mock_name = 'uint16'
mock_np.uint8 = MagicMock()
mock_np.uint8._mock_name = 'uint8'
mock_np.int16 = MagicMock()
mock_np.int16._mock_name = 'int16'

mock_np.ndarray = FakeNDArray

def fake_dtype_factory(x):
    m = MagicMock()
    # If x is a MagicMock, we can't rely on str(x) to contain the type name
    # unless we specifically set its name or it's a specific mock.
    # Let's try to match by identity if possible, but they are defined in convert_utils
    # which is loaded AFTER this factory is assigned to mock_np.dtype.

    # Let's use a simpler approach: use a closure or global to track what was requested.
    # Or even better, just return an object that HAS an itemsize property that we can set.

    class DTypeMock:
        def __init__(self, name):
            # Try to get name from MagicMock if that's what we got
            if hasattr(name, '_mock_name') and name._mock_name:
                self._name = name._mock_name.lower()
            else:
                self._name = str(name).lower()

            if '16' in self._name or 'int16' in self._name or 'uint16' in self._name or 'float16' in self._name:
                self.itemsize = 2
            elif '8' in self._name or 'int8' in self._name or 'uint8' in self._name:
                self.itemsize = 1
            else:
                self.itemsize = 4
    return DTypeMock(x)

mock_np.dtype = fake_dtype_factory
sys.modules["numpy"] = mock_np

# --- Module Loading ---
# Using importlib to follow project patterns for modules with potential dependency issues
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convert_utils.py')
spec = importlib.util.spec_from_file_location("convert_utils", module_path)
convert_utils = importlib.util.module_from_spec(spec)
sys.modules["convert_utils"] = convert_utils
spec.loader.exec_module(convert_utils)

class TestConvertUtils(unittest.TestCase):
    """Unit tests for convert_utils.py"""

    def test_permute_standard(self):
        """Verify permute logic with n_head == n_head_kv (standard case)."""
        # Input: 8 elements, shape (8, 1), 2 heads
        # Logic: reshape(2, 2, 2, 1) -> swapaxes(1, 2) -> (2, 2, 2, 1) -> reshape(8, 1)
        data = [0, 1, 2, 3, 4, 5, 6, 7]
        weights = FakeNDArray(data, (8, 1))

        result = convert_utils.permute(weights, n_head=2, n_head_kv=2)

        self.assertEqual(result.shape, (8, 1))
        # Expected interleaving (RoPE style): [0, 2, 1, 3, 4, 6, 5, 7]
        self.assertEqual(result.data, [0, 2, 1, 3, 4, 6, 5, 7])

    def test_permute_mismatched_heads(self):
        """Verify permute logic when n_head != n_head_kv (GQA case)."""
        data = [0, 1, 2, 3, 4, 5, 6, 7]
        weights = FakeNDArray(data, (8, 1))

        # Should prioritize n_head_kv=2 over n_head=4
        result = convert_utils.permute(weights, n_head=4, n_head_kv=2)

        self.assertEqual(result.shape, (8, 1))
        self.assertEqual(result.data, [0, 2, 1, 3, 4, 6, 5, 7])

    def test_permute_multi_dim(self):
        """Verify permute logic with higher dimensional weights."""
        # 16 elements, shape (8, 2), 2 heads
        # Head 0: elements 0-7, Head 1: elements 8-15
        data = list(range(16))
        weights = FakeNDArray(data, (8, 2))

        result = convert_utils.permute(weights, n_head=2, n_head_kv=2)

        self.assertEqual(result.shape, (8, 2))
        # For first head: [0,1, 4,5, 2,3, 6,7]
        expected_head0 = [0, 1, 4, 5, 2, 3, 6, 7]
        expected_head1 = [8, 9, 12, 13, 10, 11, 14, 15]
        self.assertEqual(result.data, expected_head0 + expected_head1)

    def test_data_type_elements_to_bytes(self):
        """Verify DataType.elements_to_bytes correctly calculates size."""
        # DT_F32 is defined in convert_utils.py
        dt_f32 = convert_utils.DT_F32
        self.assertEqual(dt_f32.elements_to_bytes(10), 40)

        # DT_F16 is defined in convert_utils.py
        dt_f16 = convert_utils.DT_F16
        self.assertEqual(dt_f16.elements_to_bytes(10), 20)

    def test_quantized_data_type_elements_to_bytes(self):
        """Verify QuantizedDataType handles block-based sizing and constraints."""
        dt_q8_0 = convert_utils.DT_Q8_0

        # Q8_0 uses 32-element blocks.
        # Real Q8_0 itemsize: f16 (2) + 32*i1 (32) = 34 bytes per block.
        with patch.object(dt_q8_0.quantized_dtype, 'itemsize', 34):
            self.assertEqual(dt_q8_0.elements_to_bytes(64), 68)

        # Should raise error if not a multiple of block size
        with self.assertRaises(AssertionError):
            dt_q8_0.elements_to_bytes(31)

if __name__ == "__main__":
    unittest.main()
