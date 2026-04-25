import torch
import torch.nn.functional as F
import time

@torch.compile
def relu_sq_1(x):
    return F.relu(x) ** 2

@torch.compile
def relu_sq_2(x):
    r = F.relu(x)
    return r * r

@torch.compile
def relu_sq_3(x):
    return x * F.relu(x)

x = torch.randn(10000, 6912, device='cpu')

# Warmup
for _ in range(10):
    relu_sq_1(x)
    relu_sq_2(x)
    relu_sq_3(x)

t0 = time.time()
for _ in range(100):
    relu_sq_1(x)
print("relu ** 2:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    relu_sq_2(x)
print("relu * relu:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    relu_sq_3(x)
print("x * relu:", time.time() - t0)
