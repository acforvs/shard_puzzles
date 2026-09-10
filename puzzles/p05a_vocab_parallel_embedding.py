"""Puzzle 05a - Vocab-parallel embedding (tensor + sequence parallel)

    y = E[ids]          E: [v d]   (v, the vocabulary, is often 100k+ rows)

Mesh tp.  ids: [b s], replicated integers in [0, v).  Return y: [b s:tp d], sequence-parallel,
ready for the first block of 02a/03a.

The table is too big to replicate, so split it by rows. A rank can only look up the ids it owns.
What does it put in the other positions, and which ONE collective gives the output layout?

embedding(ids, table) does the masked local lookup.

Read: Megatron-LM Sec. 3 (input embedding); Ultra-Scale Playbook, "Sequence Parallelism".
"""

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class VocabParallelEmbedding(Module):
    def on_init(self, p):
        # TODO: split the table
        self.E = TODO

    def forward(self, ids):
        ids = ids.expect("b s")  # token ids, replicated
        # TODO: look up what this rank owns, then one collective
        y = TODO
        self.ids = ids  # saved for the backward (05b)
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s v d",
    params={"E": "v d"},
    inputs={"ids": Ids("b s", high="v")},
    output="b s:tp d",
    configs=[
        {"b": 2, "s": 4, "v": 12, "d": 6, "tp": 2},
        {"b": 1, "s": 8, "v": 16, "d": 4, "tp": 4},
        {"b": 3, "s": 6, "v": 9, "d": 5, "tp": 3},
    ],
    reference=lambda c, p, ids: p.E[ids],
    max_weight_elements='v*d/tp',
    max_forward_collectives=1,
    max_comm_bytes="2*b*s*d*(tp-1)/tp",
)
