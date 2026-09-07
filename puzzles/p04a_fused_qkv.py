"""Puzzle 04a - Fused QKV projection under tensor parallelism

The attention of 03a, but forward may multiply by weights only TWICE: one fused QKV
projection and the output projection.

Mesh tp.  x: [b s:tp d].  Return [b s:tp d].  h and hkv are divisible by tp, h > hkv.

The obvious fusion is cat([Wq, Wk, Wv]) followed by a column split. The shapes all line
up and the numbers come out wrong. Work out which heads rank 0 actually holds, then build
a fused layout in which every rank's contiguous slice holds its q heads AND their k and v.

Read: Megatron-LM's attention (`get_query_key_value_tensors`, query-group-major QKV layout).
"""

from puzzles.p03a_attention_tp_sp import SPEC as UNFUSED
from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class FusedAttention(Module):
    def on_init(self, p):
        # TODO: build Wqkv so that each rank's contiguous column slice is [its q heads | their k | their v]
        wq = TODO
        wk = TODO
        wv = TODO
        wqkv = TODO
        self.Wqkv = TODO
        self.Wo = shard(p.Wo, "(h*hd):tp d")

    def forward(self, x):
        x = x.expect("b s:tp d")
        x = all_gather(x, "s", tp).expect("b s d")
        # TODO: one projection, then split it into q [b h:tp s hd] and k, v [b hkv:tp s hd]
        qkv = TODO
        q, k, v = TODO
        q = TODO
        k = TODO
        v = TODO
        k, v = repeat_interleave(k, "h/hkv", dim=1), repeat_interleave(v, "h/hkv", dim=1)
        scores = (q @ k.transpose(-1, -2)).expect("b h:tp s s")
        probs = softmax(scores * size("hd") ** -0.5 + causal_mask("s s"), dim=-1)
        o = (probs @ v).expect("b h:tp s hd").transpose(1, 2).view("b s (h*hd):tp")
        self.x, self.q, self.k, self.v, self.probs, self.o = x, q, k, v, probs, o  # saved for 04b
        y = (o @ self.Wo).expect("b s d", partial=tp)
        return reduce_scatter(y, "s", tp).expect("b s:tp d")


# ---- puzzle definition -------------------------------------------------------
SPEC = UNFUSED.replace(
    configs=[
        {"b": 2, "s": 6, "d": 16, "h": 4, "hkv": 2, "hd": 4, "tp": 2},
        {"b": 1, "s": 8, "d": 12, "h": 8, "hkv": 4, "hd": 3, "tp": 4},
        {"b": 2, "s": 4, "d": 8, "h": 12, "hkv": 4, "hd": 2, "tp": 2},
    ],
    max_weight_matmuls=2,
)
