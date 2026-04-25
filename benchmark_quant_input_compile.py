import torch
import time

@torch.compile
def quant_input_1(input):
    s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (input * s).round().clamp(-128, 127).to(torch.int8), s

@torch.compile
def quant_input_2(input):
    s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return input.mul(s).round_().clamp_(-128, 127).to(torch.int8), s

x = torch.randn(10000, 6912, device='cpu', dtype=torch.bfloat16)

for _ in range(10):
    x1 = x.clone()
    quant_input_1(x1)
    x2 = x.clone()
    quant_input_2(x2)

t0 = time.time()
for _ in range(100):
    x1 = x.clone()
    quant_input_1(x1)
print("original:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    x2 = x.clone()
    quant_input_2(x2)
print("optimized:", time.time() - t0)
