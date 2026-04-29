import unittest
from unittest.mock import patch
import torch

with patch('ctypes.CDLL'):
    from gpu.model import squared_relu

class TestGPUModel(unittest.TestCase):
    def test_squared_relu_positive(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([1.0, 4.0, 9.0])
        result = squared_relu(x)
        torch.testing.assert_close(result, expected)

    def test_squared_relu_negative(self):
        x = torch.tensor([-1.0, -2.0, -3.0])
        expected = torch.tensor([0.0, 0.0, 0.0])
        result = squared_relu(x)
        torch.testing.assert_close(result, expected)

    def test_squared_relu_mixed_and_zero(self):
        x = torch.tensor([-2.0, 0.0, 2.0, -0.5, 0.5])
        expected = torch.tensor([0.0, 0.0, 4.0, 0.0, 0.25])
        result = squared_relu(x)
        torch.testing.assert_close(result, expected)

if __name__ == '__main__':
    unittest.main()
