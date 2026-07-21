"""
train.py — train an MLP on the dementia data using ONLY our own autograd engine.

The training loop is the same four steps every neural network uses, including
the giant ones Cohere trains:
  1. forward pass  -> predictions
  2. loss          -> one number measuring wrongness
  3. backward pass -> every weight's gradient (our engine does this)
  4. update        -> nudge each weight opposite its gradient (gradient descent)
"""

import random
from engine import Value
from nn import MLP
import data


def predict_prob(model, x):
    """
    Model outputs tanh in (-1, 1). Map it to a probability in (0, 1)
    with (out + 1) / 2 so we can do binary classification.
    """
    out = model(x)
    return (out + 1) * 0.5


def bce_loss(prob, y):
    """
    Binary cross-entropy for one example.
    y is 0 or 1. eps keeps the log away from 0.
    loss = -[ y*log(p) + (1-y)*log(1-p) ]
    """
    eps = 1e-7
    p = prob if isinstance(prob, Value) else Value(prob)
    # clamp via simple arithmetic to stay in (eps, 1-eps)
    p = p * (1 - 2 * eps) + eps
    return -(y * p.log_() + (1 - y) * (1 - p).log_())


# our engine needs a log; add it here to keep engine.py focused on the core.
import math
def _log(self):
    out = Value(math.log(self.data), (self,), 'log')
    def _backward():
        self.grad += (1.0 / self.data) * out.grad   # d/dx log(x) = 1/x
    out._backward = _backward
    return out
Value.log_ = _log


def accuracy(model, dataset):
    correct = 0
    for x, y in dataset:
        p = predict_prob(model, x).data
        pred = 1 if p >= 0.5 else 0
        correct += (pred == y)
    return correct / len(dataset)


def main():
    random.seed(42)
    train_set, test_set = data.load()
    print(f"train examples: {len(train_set)} | test examples: {len(test_set)}")
    print(f"features: {data.FEATURE_NAMES}")

    # 6 inputs -> hidden layer of 8 -> hidden layer of 8 -> 1 output
    model = MLP([6, 8, 8, 1])
    print(f"total parameters: {len(model.parameters())}\n")

    lr = 0.05
    epochs = 40

    for epoch in range(1, epochs + 1):
        # ---- one full pass over the training data ----
        total_loss = Value(0.0)
        for x, y in train_set:
            prob = predict_prob(model, x)     # 1) forward
            total_loss = total_loss + bce_loss(prob, y)   # 2) accumulate loss
        total_loss = total_loss * (1.0 / len(train_set))  # mean loss

        model.zero_grad()                     # clear old grads
        total_loss.backward()                 # 3) backward: fills every weight's .grad

        for p in model.parameters():          # 4) update: gradient descent step
            p.data -= lr * p.grad

        if epoch == 1 or epoch % 5 == 0:
            tr = accuracy(model, train_set)
            te = accuracy(model, test_set)
            print(f"epoch {epoch:2d} | loss {total_loss.data:.4f} "
                  f"| train acc {tr:.3f} | test acc {te:.3f}")

    print("\nDone. The loss dropped and accuracy rose using gradients")
    print("computed entirely by the autograd engine you built.")


if __name__ == "__main__":
    main()
