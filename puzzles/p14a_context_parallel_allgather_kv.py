"""Puzzle 14a - Context parallelism with all-gathered K/V

The causal GQA attention of 03a, with the SEQUENCE split for long context and every weight
replicated:  x: [b s:cp d] -> [b s:cp d].  Rank r holds positions r*s/cp ... (r+1)*s/cp - 1.

Every query needs every earlier key, so each rank must see K and V for the whole sequence.
Queries stay local. Gather as little as you can, then mask with causal_mask("s:cp s"), which
knows each rank's query positions.

With this contiguous split, which rank does the most attention work? (Llama 3 balances it by
giving each rank two chunks, one early and one late.)

Compare with 13a: does cp still have to divide the number of KV heads?

Read: Llama 3 (arXiv 2407.21783) Sec. 3.3.2; Ultra-Scale Playbook, "Context Parallelism".
"""

from puzzles.p03a_attention_tp_sp import ATTENTION_PARAMS
from shardpuzzles import *
from shardpuzzles.ref import gqa_attention


# ---- your code ---------------------------------------------------------------
class AllGatherKVAttention(Module):
    def on_init(self, p):
        self.Wq, self.Wk, self.Wv, self.Wo = p.Wq, p.Wk, p.Wv, p.Wo  # replicated

    def forward(self, x):
        x = x.expect("b s:cp d")
        # TODO: project this rank's tokens into q [b h s:cp hd], k and v [b hkv s:cp hd]
        q = TODO
        k = TODO
        v = TODO
        # TODO: give every rank the whole sequence of keys and values (as few bytes as possible)
        k = TODO
        v = TODO
        k, v = repeat_interleave(k, "h/hkv", dim=1), repeat_interleave(v, "h/hkv", dim=1)
        # TODO: attention for this rank's queries; then o as one row per token, ready for Wo
        scores = TODO
        probs = TODO
        o = TODO
        return (o @ self.Wo).expect("b s:cp d")


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d h hkv hd",
    params=ATTENTION_PARAMS,
    inputs={"x": "b s:cp d"},
    output="b s:cp d",
    configs=[
        {"b": 2, "s": 8, "d": 16, "h": 4, "hkv": 2, "hd": 4, "cp": 2},
        {"b": 1, "s": 12, "d": 12, "h": 6, "hkv": 2, "hd": 3, "cp": 4},
        {"b": 2, "s": 9, "d": 8, "h": 2, "hkv": 1, "hd": 4, "cp": 3},
        {"b": 1, "s": 4, "d": 8, "h": 4, "hkv": 4, "hd": 2, "cp": 2},
        {"b": 1, "s": 1, "d": 6, "h": 2, "hkv": 1, "hd": 2, "cp": 1},
    ],
    reference=gqa_attention,
    require_bytes=True,
    max_forward_collectives=2,
    max_comm_bytes="4*b*hkv*s*hd*(cp-1)/cp",
)
