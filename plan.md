1. **Refactor `preprocess_two_weights` in `3rdparty/llama.cpp/test_op/preprocess.py` and `3rdparty/llama.cpp/test_op/preprocess_mad_lut.py`**
   - Replace the deeply nested `for` loops (which slice via `np.split` and reassemble via `np.concatenate`) with zero-copy NumPy spatial views (`reshape`, `transpose`).
   - This prevents multiple intermediate multi-megabyte array allocations, speeding up the block formatting layout natively in C.
   - Specifically, implement:
     ```python
     w_blocks = weight.reshape(
         M // BM, BM // bm, bm,
         K // BY, BY // by, by // 2
     ).transpose(0, 3, 1, 4, 2, 5).reshape(-1, bm, by // 2)
     left = w_blocks[:, :, 0].reshape(-1, 4, bm // 4)[:, [0, 2, 1, 3], :].reshape(-1, bm)
     right = w_blocks[:, :, 1].reshape(-1, 4, bm // 4)[:, [0, 2, 1, 3], :].reshape(-1, bm)
     ```
     then concatenate and append to `final_weight`.

2. **Refactor `preprocess_three_weights` in `3rdparty/llama.cpp/test_op/preprocess.py` and `3rdparty/llama.cpp/test_op/preprocess_mad_lut.py`**
   - Apply the same O(1) zero-copy spatial views as above to the weight matrix.
   - For `sign_weight`, apply `np.abs(weight, out=weight)` instead of `weight = np.abs(weight)` to avoid an implicit multi-megabyte memory reallocation.
   - Optimize the ternary encoding bits by replacing the 8-loop slicing logic on the `sign_weight` with vectorized bitwise shifts and sums:
     ```python
     sign_w_blocks = sign_weight.reshape(
         M // BM, BM // bm, bm,
         K // BY, BY // (by * 4), (by * 4) // 3
     ).transpose(0, 3, 1, 4, 2, 5).reshape(-1, bm, 8)
     sign_split = sign_w_blocks.reshape(-1, 2, 16, 8)
     top = sign_split[:, 0, :, :].astype(np.uint16)
     bot = sign_split[:, 1, :, :].astype(np.uint16)
     shifts_top = 15 - (2 * np.arange(8))
     shifts_bot = 15 - (2 * np.arange(8) + 1)
     combine_weight = (top << shifts_top).sum(axis=-1) + (bot << shifts_bot).sum(axis=-1)
     ```

3. **Complete Pre-Commit Steps.**
   - Call the `pre_commit_instructions` tool to run the necessary verification, testing, formatting, and linting checks according to the project's instructions.
   - Update Hugo's `.jules/hugo.md` journal with the critical learning about replacing nested slicing loops with O(1) spatial view transpositions.

4. **Submit changes in branch `hugo-optimization`.**
