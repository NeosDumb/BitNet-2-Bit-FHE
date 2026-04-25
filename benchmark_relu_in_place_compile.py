import torch
import torch.nn.functional as F
import time

@torch.compile
def relu_sq_1(x):
    return F.relu(x) ** 2

@torch.compile
def relu_sq_2(x):
    return F.relu(x).pow(2)

@torch.compile
def relu_sq_3(x):
    r = F.relu(x)
    return r.mul(r)

x = torch.randn(10000, 6912, device='cpu')

# Warmup
for _ in range(10):
    x1 = x.clone()
    relu_sq_1(x1)
    x2 = x.clone()
    relu_sq_2(x2)
    x3 = x.clone()
    relu_sq_3(x3)


t0 = time.time()
for _ in range(100):
    x1 = x.clone()
    relu_sq_1(x1)
print("relu ** 2:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x2 = x.clone()
    relu_sq_2(x2)
print("relu.pow(2):", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x3 = x.clone()
    relu_sq_3(x3)
print("relu.mul(r):", time.time() - t0)
