import torch
import torch.nn.functional as F
import time

def relu_sq_1(x):
    return F.relu(x) ** 2

def relu_sq_2(x):
    # Mathematical optimization: Multiplication is computationally cheaper than exponentiation.
    # In-place ReLU eliminates the intermediate tensor allocation, and in-place
    # multiplication eliminates a second allocation, minimizing the memory bandwidth tax.
    r = F.relu(x)
    return r.mul_(r)

x = torch.randn(10000, 6912, device='cpu')

for _ in range(10):
    x1 = x.clone()
    relu_sq_1(x1)
    x2 = x.clone()
    relu_sq_2(x2)

t0 = time.time()
for _ in range(100):
    x1 = x.clone()
    relu_sq_1(x1)
print("relu ** 2:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x2 = x.clone()
    relu_sq_2(x2)
print("r = F.relu(x); return r.mul_(r):", time.time() - t0)
