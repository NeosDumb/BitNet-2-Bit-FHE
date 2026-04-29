import unittest
import torch
import sys
import os

# Add the parent directory to the path so that we can import gpu modules directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu.sample_utils import top_p

class TestSampleUtils(unittest.TestCase):
    def test_top_p_deterministic(self):
        # Create a prob distribution where one token clearly dominates
        probs = torch.tensor([[0.9, 0.05, 0.03, 0.02]])

        # With p=0.8, only the first token (index 0, prob 0.9) should be considered.
        for _ in range(10):
            token = top_p(probs, 0.8)
            self.assertEqual(token.item(), 0)

    def test_top_p_uniform(self):
        # Create uniform probability distribution
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

        # Set a seed for reproducibility
        torch.manual_seed(42)

        # p=1.0 means all tokens should be possible
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(100):
            token = top_p(probs, 1.0)
            counts[token.item()] += 1

        # All tokens should be sampled at least once
        for k, v in counts.items():
            self.assertGreater(v, 0)

    def test_top_p_threshold(self):
        # Sort is descending: 0.5, 0.3, 0.1, 0.1
        probs = torch.tensor([[0.1, 0.5, 0.3, 0.1]])

        # If p=0.6, the mask keeps the first two elements: 0.5 (idx 1), 0.3 (idx 2)
        # We can verify it only samples 1 or 2
        torch.manual_seed(42)
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(50):
            token = top_p(probs, 0.6)
            counts[token.item()] += 1

        self.assertEqual(counts[0], 0)
        self.assertGreater(counts[1], 0)
        self.assertGreater(counts[2], 0)
        self.assertEqual(counts[3], 0)

    def test_top_p_single_token(self):
        # Edge case: vocabulary of size 1
        probs = torch.tensor([[1.0]])
        token = top_p(probs, 0.5)
        self.assertEqual(token.item(), 0)

    def test_top_p_batch(self):
        # Test with a batch size > 1
        probs = torch.tensor([
            [0.9, 0.1, 0.0, 0.0],  # mostly token 0
            [0.1, 0.0, 0.9, 0.0]   # mostly token 2
        ])

        torch.manual_seed(42)
        for _ in range(10):
            tokens = top_p(probs, 0.5)
            self.assertEqual(tokens.shape, (2, 1))
            self.assertEqual(tokens[0, 0].item(), 0)
            self.assertEqual(tokens[1, 0].item(), 2)

    def test_top_p_2d_output(self):
         probs = torch.tensor([[0.8, 0.2]])
         token = top_p(probs, 0.5)
         self.assertEqual(token.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
