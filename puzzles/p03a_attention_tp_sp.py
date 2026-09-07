"""Puzzle 03a - Causal GQA attention with tensor + sequence parallelism

    q = x Wq (h heads of size hd)      k = x Wk, v = x Wv (hkv heads, each shared by h/hkv query heads)
    o = softmax(q k^T / sqrt(hd) + causal mask) v           y = o Wo
    Wq: [d h*hd]   Wk, Wv: [d hkv*hd]   Wo: [h*hd d]       (columns are head-major)

Mesh tp.  x: [b s:tp d].  Return [b s:tp d].  h and hkv are divisible by tp.

Heads are independent, so split them across ranks. Which query heads does a rank own, and
which kv heads does it need for them? Which tensor is the largest activation?

Helpers: causal_mask("s s") (the additive mask), size("hd") (a size as a number), softmax(t, dim).

Read: Megatron-LM (arXiv 1909.08053) Sec. 3, Fig. 3b; GQA (arXiv 2305.13245).
"""

from shardpuzzles import *
from shardpuzzles.ref import gqa_attention


# ---- your code ---------------------------------------------------------------
class Attention(Module):
    def on_init(self, p):
        # TODO: split the projections by heads
        self.Wq = TODO
        self.Wk = TODO
        self.Wv = TODO
        self.Wo = TODO

    def forward(self, x):
        x = x.expect("b s:tp d")
        # TODO: gather the sequence, then project q, k, v into this rank's heads, as [b heads s hd]
        x = TODO
        q = TODO
        k = TODO
        v = TODO
        k = repeat_interleave(k, "h/hkv", dim=1)  # each kv head serves h/hkv query heads
        v = repeat_interleave(v, "h/hkv", dim=1)
        # TODO: causal attention on this rank's heads; then o as one row per token, ready for Wo
        scores = TODO
        probs = TODO
        o = TODO
        # TODO: output projection, back to sequence-parallel
        y = TODO
        self.x, self.q, self.k, self.v, self.probs, self.o = x, q, k, v, probs, o  # saved for 03b
        return y


# ---- puzzle definition -------------------------------------------------------
ATTENTION_PARAMS = {"Wq": "d h*hd", "Wk": "d hkv*hd", "Wv": "d hkv*hd", "Wo": "h*hd d"}

SPEC = Spec(
    dims="b s d h hkv hd",
    params=ATTENTION_PARAMS,
    inputs={"x": "b s:tp d"},
    output="b s:tp d",
    configs=[
        {"b": 2, "s": 6, "d": 16, "h": 4, "hkv": 2, "hd": 4, "tp": 2},
        {"b": 1, "s": 8, "d": 12, "h": 8, "hkv": 4, "hd": 3, "tp": 4},
        {"b": 2, "s": 4, "d": 8, "h": 4, "hkv": 4, "hd": 2, "tp": 2},
        {"b": 1, "s": 6, "d": 8, "h": 12, "hkv": 4, "hd": 2, "tp": 2},
    ],
    reference=gqa_attention,
    max_weight_elements='(2*d*h*hd+2*d*hkv*hd)/tp',
    max_forward_collectives=2,
)
