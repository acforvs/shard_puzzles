"""Puzzle 00c - Data parallelism: a linear layer, forward and backward

    y = x @ W        x: [b:dp s d] (each dp rank has its own micro-batch)    W: [d f], replicated

Now write both a forward pass and its backward pass.
The checker feeds backward dy = dL/dy, laid out like forward's output, and compares what you return
with autograd on the single-device reference.

forward(x) -> y [b:dp s f]. No communication: every rank has all of W.

backward(dy) -> {"x": dx [b:dp s d], "W": dW [d f]}, where dy = dL/dy is laid out like y.
  * dx = dy @ W^T (`W.transpose(0, 1)`), one row per token.
  * Every token used the same W, so W's gradient is a SUM over tokens:
        dW[d, f] = sum_b sum_s x[b, s, d] * dy[b, s, f]  =  einsum("b s d, b s f -> d f", x, dy)
    einsum labels each operand's dims in order and sums the labels missing from the output.
    b is split over dp, so each rank only summed its own tokens: a partial sum over dp (00b's
    rule, with b as the summed dim). One all_reduce makes it whole. That is data parallelism (DDP).

Read: PyTorch DDP (arXiv 2006.15704) Sec. 2; Ultra-Scale Playbook, "Data Parallelism".
"""

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class Linear(Module):
    def on_init(self, p):
        self.W = p.W  # replicated: every dp rank holds all of W

    def forward(self, x):
        x = x.expect("b:dp s d")  # the input: this rank's share of the batch
        self.x = x  # saved for backward
        # TODO: compute y and state its layout
        y = TODO
        return y

    def backward(self, dy):
        dy = dy.expect("b:dp s f")  # the gradient arrives laid out like forward's output
        # TODO: dx needs no communication
        dx = TODO
        # TODO: dW: an einsum over this rank's tokens, then make it whole (bytes: an all-reduce sends 2*2*numel*(g-1)/g)
        dW = TODO
        return {"x": dx, "W": dW}


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d f",
    params={"W": "d f"},
    inputs={"x": "b:dp s d"},
    output="b:dp s f",
    configs=[
        {"b": 4, "s": 3, "d": 8, "f": 6, "dp": 2},
        {"b": 8, "s": 2, "d": 4, "f": 5, "dp": 4},
        {"b": 6, "s": 2, "d": 3, "f": 7, "dp": 3},
    ],
    reference=lambda c, p, x: x @ p.W,
    grads={"x": "b:dp s d", "W": "d f"},
    require_bytes=True,
    max_forward_collectives=0,
)
