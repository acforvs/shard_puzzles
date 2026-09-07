"""Puzzle 03b - Backward: causal GQA attention with tensor + sequence parallelism

The forward is 03a's reference solution, imported below. It saves self.x (gathered), self.q / self.k /
self.v ([b h:tp s hd]; k, v repeated), self.probs and self.o ([b s (h*hd):tp], the input to Wo).
dy: [b s:tp d]. Return
    {"x": [b s:tp d], "Wq": [d (h*hd):tp], "Wk": [d (hkv*hd):tp], "Wv": [d (hkv*hd):tp], "Wo": [(h*hd):tp d]}.

Attention backward, from the saved probabilities P (the per-head math is given):
    dV = P^T dO     dP = dO V^T     dS = softmax_backward(dP, P) * scale     dQ = dS K     dK = dS^T Q
Each kv head was repeated h/hkv times. What's the backward of a repeat, and where is communication needed?

Helpers: softmax_backward, repeat_interleave_backward(t, n, dim) (sums each group of n heads).

Read: FlashAttention-2 (arXiv 2307.08691) Sec. 2.2.
"""

from puzzles.p03a_attention_tp_sp import SPEC as FORWARD
from shardpuzzles import *
from solutions.p03a_attention_tp_sp import Attention  # 03a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class AttentionBackward(Attention):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        # TODO: undo the reduce_scatter; Wo's gradient; dO per head as [b h:tp s hd]
        dy = TODO
        dWo = TODO
        do = TODO
        dv = (self.probs.transpose(-1, -2) @ do).expect("b h:tp s hd")
        dprobs = (do @ self.v.transpose(-1, -2)).expect("b h:tp s s")
        ds = softmax_backward(dprobs, self.probs, dim=-1) * size("hd") ** -0.5
        dq = (ds @ self.k).expect("b h:tp s hd")
        dk = (ds.transpose(-1, -2) @ self.q).expect("b h:tp s hd")
        # TODO: undo the kv repeat, then back to [b s (heads*hd):tp]
        dk = TODO
        dv = TODO
        dq = TODO
        # TODO: weight gradients and dx
        dWq = TODO
        dWk = TODO
        dWv = TODO
        dx = TODO
        return {"x": dx, "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(
    grads={
        "x": "b s:tp d",
        "Wq": "d (h*hd):tp",
        "Wk": "d (hkv*hd):tp",
        "Wv": "d (hkv*hd):tp",
        "Wo": "(h*hd):tp d",
    }
)
