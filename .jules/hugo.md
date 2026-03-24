## 2024-05-24 - Sign Invariance in Scale-Based Quantization
**Learning:** In scale-based quantization routines, the condition `val * scale > 0` mathematically simplifies to `val > 0` since the global maximum scale is strictly non-negative. This sign invariance decouples the quantization of individual values from the global max-scan.
**Action:** Apply fused-loop optimization to perform max-scan, quantization, and bit-packing in a single pass. This eliminates intermediate heap allocations (O(1) memory mapping), reduces memory traffic, and speeds up the entire quantization process.
