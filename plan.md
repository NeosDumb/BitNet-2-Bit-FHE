1. **Apply the patch:** Run `python3 patch_neon.py` to apply the SIMD mathematical reduction optimizations in `src/ggml-bitnet-mad.cpp`.
2. **Build and Test Performance:** I will build the project using `cmake -B build && cmake --build build` to verify compilation, and run `bash utils/test_gemm_kernel.sh` to check the performance metrics.
3. **Run tests:** Run the full test suite using `python3 -m unittest discover -s . -p 'test_*.py'` to ensure no regressions were introduced.
4. **Complete pre commit steps:** Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.
5. **Submit the change:** Once everything looks good, I will push the PR with a descriptive title and message using the `submit` tool.
