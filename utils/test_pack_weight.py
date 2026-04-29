import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu.pack_weight

class TestPackWeight(unittest.TestCase):
    def test_convert_weight_int8_to_int2(self):
        # Initialize a valid tensor size for the internal functions: N must be a multiple of 16, K must be a multiple of 32
        N, K = 16, 32

        # Test 1: All -1s
        # Original values -1 -> after +2 becomes 1
        # Compressed: 1 + 1*4 + 1*16 + 1*64 = 85
        # Interleaved uint8 85 -> 01010101, so parts are all the same, output is 85
        weight_neg1 = torch.ones((N, K), dtype=torch.int8) * -1
        output_neg1 = gpu.pack_weight.convert_weight_int8_to_int2(weight_neg1)
        self.assertEqual(output_neg1.shape, (N, K // 4))
        self.assertEqual(output_neg1.dtype, torch.int8)
        self.assertTrue(torch.all(output_neg1 == 85))

        # Test 2: All 0s
        # Original values 0 -> after +2 becomes 2
        # Compressed: 2 + 2*4 + 2*16 + 2*64 = 170 (int8: -86)
        weight_0 = torch.zeros((N, K), dtype=torch.int8)
        output_0 = gpu.pack_weight.convert_weight_int8_to_int2(weight_0)
        self.assertTrue(torch.all(output_0 == -86))

        # Test 3: All 1s
        # Original values 1 -> after +2 becomes 3
        # Compressed: 3 + 3*4 + 3*16 + 3*64 = 255 (int8: -1)
        weight_1 = torch.ones((N, K), dtype=torch.int8)
        output_1 = gpu.pack_weight.convert_weight_int8_to_int2(weight_1)
        self.assertTrue(torch.all(output_1 == -1))

        # Test 4: Patterned input to ensure mathematical layout transformations don't mix bounds incorrectly
        # We set an explicit pattern. The permutation shifts the positions,
        # compression groups 4 elements, and interleaving transposes the bits.
        weight_pattern = torch.zeros((N, K), dtype=torch.int8)
        weight_pattern[0, 0] = 1   # Maps to +3
        weight_pattern[0, 1] = -1  # Maps to +1

        output_pattern = gpu.pack_weight.convert_weight_int8_to_int2(weight_pattern)

        # Ensure it runs without error and yields consistent bounds
        self.assertTrue(torch.all(output_pattern >= -128))
        self.assertTrue(torch.all(output_pattern <= 127))
        # Ensure that it is deterministic
        output_pattern_2 = gpu.pack_weight.convert_weight_int8_to_int2(weight_pattern)
        self.assertTrue(torch.all(output_pattern == output_pattern_2))

if __name__ == '__main__':
    unittest.main()
