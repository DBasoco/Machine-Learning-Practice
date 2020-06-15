# learning how neural networks actually are coded

import torch
import numpy as np

x = torch.empty(5, 3) # 5x3 matrix, uninitialized

x = torch.rand(5, 3) # randomly initialized

x = torch.zeros(5, 3, dtype=torch.long) # filled with zeros, of dtype long

x = torch.tensor([5.5, 3]) # tensor directly from data
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float) # make tensors like existing tensors
print(x)

print(x.size()) # size is a tuple and returns size

y = torch.rand(5, 3) # s.t. they can be added
print(x + y)
print(torch.add(x, y)) # another way to add

result = torch.empty(5, 3)
torch.add(x, y, out=result) # output tensor as argument
print(result)

y.add_(x) # addition in place
print(y)

print(x[:, 1]) # torch uses standard numpy indexing


# resize using torch.view
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# if you hae a one element tensor use .item() to get python number
x = torch.rand(1)
print(x)
print(x.item())

# convert a tensor to an array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
