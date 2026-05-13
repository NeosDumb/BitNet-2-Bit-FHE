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

    # Mathematical Optimization: Conservation of Memory via In-Place Operations
    # The evaluation of `mask = probs_sum - probs_sort > p` explicitly allocates an intermediate
    # float32 subtraction tensor across the vocabulary dimension before performing the comparison,
    # incurring an O(N) memory allocation "energy tax".
    # While algebraic substitution `probs_sum > p + probs_sort` avoids tensor-tensor subtraction,
    # PyTorch still allocates an intermediate tensor for the scalar-tensor addition `p + probs_sort`.
    # By modifying `probs_sum` strictly in-place via `.sub_()`, we form a closed thermodynamic system
    # that completely avoids all intermediate float32 tensor allocations.
    probs_sum.sub_(probs_sort)
    mask = probs_sum > p
    probs_sort.masked_fill_(mask, 0.0)

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token