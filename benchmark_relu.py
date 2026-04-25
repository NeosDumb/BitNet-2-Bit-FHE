import torch
import torch.nn.functional as F
import time

def relu_sq_1(x):
    return F.relu(x) ** 2

def relu_sq_2(x):
    r = F.relu(x)
    return r * r

def relu_sq_3(x):
    return x * F.relu(x)

x = torch.randn(10000, 6912, device='cpu')

# Warmup
for _ in range(10):
    relu_sq_1(x)
    relu_sq_2(x)
    relu_sq_3(x)

torch.cpu.synchronize()
t0 = time.time()
for _ in range(100):
    relu_sq_1(x)
torch.cpu.synchronize()
print("relu ** 2:", time.time() - t0)

torch.cpu.synchronize()
t0 = time.time()
for _ in range(100):
    relu_sq_2(x)
torch.cpu.synchronize()
print("relu * relu:", time.time() - t0)

torch.cpu.synchronize()
t0 = time.time()
for _ in range(100):
    relu_sq_3(x)
torch.cpu.synchronize()
print("x * relu:", time.time() - t0)
