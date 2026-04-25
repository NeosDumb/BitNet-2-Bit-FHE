import torch
import torch.nn.functional as F
import time

def relu_sq_1(x):
    return F.relu(x) ** 2

def relu_sq_2(x):
    # Mathematics: x * max(0, x) = max(0, x)^2
    return F.relu_(x).pow_(2)

def relu_sq_3(x):
    r = F.relu(x)
    return r.mul_(r)

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
print("relu_.pow_(2):", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x3 = x.clone()
    relu_sq_3(x3)
print("relu.mul_(r):", time.time() - t0)
