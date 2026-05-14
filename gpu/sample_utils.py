# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

@torch.compile
def top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): probability distribution tensor.
        p (float): probability threshold for top-p sampling.

    Returns:
        torch.Tensor: sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative
        probability mass exceeds the threshold p. The distribution is
        renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Mathematical Optimization: In-place Arithmetic for Memory Efficiency
    # Evaluating `probs_sum - probs_sort > p` allocates a full-sized intermediate
    # tensor for the subtraction result before the boolean comparison.
    # By substituting with `probs_sum.sub_(probs_sort)`, we perform the subtraction
    # in-place since `probs_sum` is not needed downstream. This entirely avoids the
    # intermediate tensor allocation 'energy tax', saving O(N) memory bandwidth per token generation step.
    mask = probs_sum.sub_(probs_sort) > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token