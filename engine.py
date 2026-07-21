"""
engine.py — a tiny autograd engine, built from scratch.

A `Value` wraps a single number. It also remembers which numbers and which
operation produced it, so we can walk that history backwards and compute
gradients (backpropagation). This is the same core idea inside PyTorch's
.backward() and JAX's grad(), stripped to its essentials.

Each operation follows the same pattern:
  1. compute the output number
  2. build a new Value that remembers its parents + operation
  3. define a local _backward() that pushes gradient to the parents
     using that operation's derivative rule
"""

import math


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data                 # the number this node holds
        self.grad = 0.0                  # d(final_output)/d(self); filled in by backward()
        self._backward = lambda: None    # how to push our grad to our parents; no-op for raw inputs
        self._prev = set(_children)      # the parent nodes that produced this one (graph edges)
        self._op = _op                   # operation name, for debugging/printing

    # ---- addition: c = a + b. local sensitivity to each parent is 1 ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad     # pass gradient straight through
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # ---- multiplication: c = a * b. sensitivity to a is b, to b is a ----
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad   # scale by the OTHER parent (chain rule)
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # ---- power: c = a**k for constant k. d/da (a**k) = k * a**(k-1) ----
    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only supports constant powers"
        out = Value(self.data ** k, (self,), f'**{k}')

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    # ---- tanh activation: squashes any number into (-1, 1). d/dx tanh = 1 - tanh^2 ----
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad     # d/dx e^x = e^x
        out._backward = _backward
        return out

    # ---- conveniences built on the primitives above ----
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    # ---- the backward pass over the whole graph ----
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0                    # nudging the output moves itself 1-for-1
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
