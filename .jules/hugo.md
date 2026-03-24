## 2025-02-18 - Sign Invariance in Scale-Based Quantization
**Learning:** In scale-based quantization (e.g., quantize_i2_s), if the scale is non-negative, the evaluation `val * scale > 0` mathematically simplifies to `val > 0`. This sign invariance enables O(1) memory mapping by eliminating temporal dependencies on the calculation of the global scale.
**Action:** When implementing scale-based quantization, fuse max-scan, quantization, and bit-packing into a single pass to eliminate intermediate memory allocations (`O(n)` to `O(1)` memory mapping).
