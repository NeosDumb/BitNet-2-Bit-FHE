import torch
import torch.nn.functional as F

x = torch.randn(10, 10)
x1 = x.clone()
x2 = x.clone()
print(F.relu(x1) ** 2)
print(F.relu_(x2).pow_(2))
print(torch.allclose(F.relu(x1) ** 2, F.relu_(x2).pow_(2)))
