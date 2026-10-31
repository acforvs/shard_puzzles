"""Puzzle 13a - Ulysses sequence parallelism (all-to-all attention)

Causal GQA attention, replicated weights, sequence split over cp: x: [b s:cp d] -> [b s:cp d].
Assume h and hkv are divisible by cp.

Re-shard q, k and v from "split by sequence" to "split by heads".
Every rank then runs ordinary full-sequence attention on h/cp heads, and you re-shard the result
back. all_to_all(t, gather_dim, scatter_dim, group) makes gather_dim whole and splits scatter_dim.

Compare the per-rank bytes with 14a's K/V all-gathers as cp grows.
Why does this head partition stop scaling once cp > hkv?

Read: DeepSpeed-Ulysses (arXiv 2309.14509) Sec. 3.
"""

from puzzles.p03a_attention_tp_sp import ATTENTION_PARAMS
from shardpuzzles import *
from shardpuzzles.ref import gqa_attention


# ---- your code ---------------------------------------------------------------
class UlyssesAttention(Module):
    def on_init(self, p):
        self.Wq, self.Wk, self.Wv, self.Wo = p.Wq, p.Wk, p.Wv, p.Wo

    def forward(self, x):
        x = x.expect("b s:cp d")
        # TODO: q, k, v for every token of this rank's heads, as [b heads s hd]
        q = TODO
        k = TODO
        v = TODO
        k, v = repeat_interleave(k, "h/hkv", dim=1), repeat_interleave(v, "h/hkv", dim=1)
        # TODO: full-sequence causal attention on this rank's heads
        scores = TODO
        probs = TODO
        o = TODO
        # TODO: back to one row per token of this rank's sequence chunk, ready for Wo
        o = TODO
        self.x, self.q, self.k, self.v, self.probs, self.o = x, q, k, v, probs, o  # saved for 13b
        return (o @ self.Wo).expect("b s:cp d")


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d h hkv hd",
    params=ATTENTION_PARAMS,
    inputs={"x": "b s:cp d"},
    output="b s:cp d",
    configs=[
        {"b": 2, "s": 8, "d": 16, "h": 4, "hkv": 2, "hd": 4, "cp": 2},
        {"b": 1, "s": 12, "d": 12, "h": 8, "hkv": 4, "hd": 3, "cp": 4},
        {"b": 2, "s": 9, "d": 8, "h": 6, "hkv": 3, "hd": 2, "cp": 3},
        {"b": 1, "s": 6, "d": 8, "h": 12, "hkv": 4, "hd": 2, "cp": 2},
        {"b": 2, "s": 4, "d": 8, "h": 4, "hkv": 4, "hd": 2, "cp": 2},
        {"b": 1, "s": 1, "d": 6, "h": 2, "hkv": 1, "hd": 2, "cp": 1},
    ],
    reference=gqa_attention,
    require_bytes=True,
    max_forward_collectives=4,
    max_comm_bytes="4*b*s*hd*(h+hkv)/cp*(cp-1)/cp",
)
