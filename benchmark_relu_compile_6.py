import torch
import torch.nn.functional as F
import time

@torch.compile
def squared_relu_original(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

@torch.compile
def squared_relu_optimized(x: torch.Tensor) -> torch.Tensor:
    # Mathematical Optimization: Multiplication vs Exponentiation
    # The ReLU operation max(0, x) outputs positive values and zeros.
    # Mathematically, multiplying the result by itself (r * r) computes the exact
    # same squared values as exponentiation (r ** 2), but multiplying natively
    # executes as a single floating-point multiplication instruction instead of
    # invoking a more complex power function routine.
    # In PyTorch, using explicit multiplication `r * r` avoids the `pow` kernel
    # dispatch overhead and memory allocation for the implicit scalar exponent `2`,
    # allowing torch.compile to fuse the operations more efficiently.
    r = F.relu(x)
    return r * r

x = torch.randn(10000, 6912, device='cpu', dtype=torch.bfloat16)

for _ in range(10):
    x1 = x.clone()
    squared_relu_original(x1)
    x2 = x.clone()
    squared_relu_optimized(x2)

t0 = time.time()
for _ in range(100):
    x1 = x.clone()
    squared_relu_original(x1)
print("original:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x2 = x.clone()
    squared_relu_optimized(x2)
print("optimized:", time.time() - t0)
