import torch
import torch.nn.functional as F
import time

@torch.compile
def relu_sq_1(x):
    return F.relu(x) ** 2

@torch.compile
def relu_sq_3(x):
    # Mathematical optimization: x * max(0, x)
    return x * F.relu(x)

x = torch.randn(10000, 6912, device='cpu')

for _ in range(10):
    x1 = x.clone()
    relu_sq_1(x1)
    x3 = x.clone()
    relu_sq_3(x3)

t0 = time.time()
for _ in range(100):
    x1 = x.clone()
    relu_sq_1(x1)
print("relu ** 2:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x3 = x.clone()
    relu_sq_3(x3)
print("x * F.relu(x):", time.time() - t0)
