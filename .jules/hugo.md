## 2024-05-18 - Scale Invariance in Quantization

**Learning:** In scale-based quantization (e.g., `quantize_i2_s`), if the scale is non-negative, the evaluation `val * scale > 0` mathematically simplifies to `val > 0`. This sign invariance enables O(1) memory mapping by eliminating temporal dependencies on the calculation of the global scale, allowing max-scan and quantization to be fused into a single pass.

**Action:** When implementing quantization kernels, analyze operations for mathematical simplifications that remove temporal dependencies, thereby enabling loop fusion and removing intermediate buffer allocations.
