from torch import nn
import torch

m = nn.Linear(2, 3) 
input = torch.randn(3, 2)
output = m(input)
print(output.size())
print(input)
print(output)