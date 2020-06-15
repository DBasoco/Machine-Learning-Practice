import torch

# grad racks all operations done on a tensor
# call method .backward() to have all gradients computed
# saved to attribute .grad
# to stop tracking use method .detach()
# prevent tracking by wrapping block in |with torch.no_grad():|

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2 # tensor operations
print(y)

# so now why has a grad_fn
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)

# torch.autograd is an engine for computing vector-Jacobian product

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# if y is no longer scalar, autograd can't compute jacobian
#  if we still want the vector product simply pass vector to backward as argument

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# stop auto_grad tracking
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# or use .detach()
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

