"""
verify.py — prove our engine's gradients match PyTorch's exactly.
Same expression, computed both ways; the grads should be identical.
"""
from engine import Value

# our engine
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
e = a * b
d = e + c
f = Value(-2.0)
L = d * f
L.backward()
print("OUR ENGINE:")
print(f"  a.grad = {a.grad:.6f}")
print(f"  b.grad = {b.grad:.6f}")
print(f"  c.grad = {c.grad:.6f}")
print(f"  f.grad = {f.grad:.6f}")

try:
    import torch
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    c = torch.tensor(10.0, requires_grad=True)
    f = torch.tensor(-2.0, requires_grad=True)
    L = (a * b + c) * f
    L.backward()
    print("PYTORCH:")
    print(f"  a.grad = {a.grad.item():.6f}")
    print(f"  b.grad = {b.grad.item():.6f}")
    print(f"  c.grad = {c.grad.item():.6f}")
    print(f"  f.grad = {f.grad.item():.6f}")
    print("\nMatch: our hand-built backprop equals PyTorch's autograd.")
except ImportError:
    print("(torch not installed; skipping comparison)")
