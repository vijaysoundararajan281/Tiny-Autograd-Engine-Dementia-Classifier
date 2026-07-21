# Tiny Autograd Engine + Dementia Classifier

A scalar-valued automatic differentiation engine built from scratch in pure
Python, with a small neural-network library on top, trained to classify
dementia from OASIS-style clinical features. No PyTorch or TensorFlow is used
for the learning itself; every gradient is computed by the engine in `engine.py`.

## Why this exists
To understand what `.backward()` actually does. The engine implements
reverse-mode autodiff (backpropagation) over a dynamically built computation
graph, the same core mechanism inside PyTorch and JAX, stripped to ~120 lines.

## Files
- `engine.py` — the `Value` autograd engine (+, *, **, tanh, exp, sub, div, backward)
- `nn.py` — `Neuron`, `Layer`, `MLP` built on `Value`
- `data.py` — OASIS-style dementia dataset (synthetic by default; real-data instructions inside)
- `train.py` — the full training loop (forward, loss, backward, update)
- `verify.py` — proves the engine's gradients match PyTorch exactly

## Run
```bash
python3 train.py     # trains the classifier, prints loss + accuracy per epoch
python3 verify.py    # checks gradients against PyTorch (needs torch installed)
```

## Result
Test accuracy ~90% on the clinical features, loss falling smoothly, with all
137 parameters' gradients computed by the hand-built engine. Verified identical
to PyTorch's autograd on a shared expression.

## Using the real OASIS data
Register at https://www.oasis-brains.org/, download the cross-sectional CSV,
then set `USE_REAL = True` and `CSV_PATH` in `data.py`. Labels: CDR == 0 →
non-demented, CDR > 0 → demented.

## What I learned
- A computation graph is just each node holding references to the parents that made it.
- Backprop = seed the output grad at 1, topologically sort, walk backward applying each op's local derivative (chain rule).
- Gradient accumulation (`+=`) is required when a value feeds multiple downstream nodes.
- Feature standardization matters a lot for training stability.
