"""
nn.py — a tiny neural-network library built on top of engine.Value.

A neuron computes  tanh( w1*x1 + w2*x2 + ... + b ).
A layer is a list of neurons. An MLP is a list of layers.
Because every operation goes through Value, calling .backward() on the loss
automatically fills in .grad for every weight in the whole network.
"""

import random
from engine import Value


class Module:
    """Base class: gives every model a way to zero grads and list its parameters."""
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_inputs):
        # one weight per input, plus one bias. small random init.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        # weighted sum of inputs + bias, then tanh
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A multi-layer perceptron: sizes = [n_in, hidden1, hidden2, ..., n_out]."""
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
