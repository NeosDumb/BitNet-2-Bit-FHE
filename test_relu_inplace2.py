import torch
import torch.nn.functional as F

x = torch.randn(10, 10)
x1 = x.clone()
x2 = x.clone()
y1 = F.relu(x1) ** 2
y2 = F.relu_(x2).pow_(2)
print((y1 - y2).abs().max())
