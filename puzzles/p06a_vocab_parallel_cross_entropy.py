"""Puzzle 06a - Vocab-parallel cross-entropy

    logits = x @ W          W: [d v]
    loss[b, s] = logsumexp(logits[b, s, :]) - logits[b, s, target[b, s]]

Mesh tp.  x: [b s d] and targets: [b s], both replicated.  Return loss: [b s], replicated.

The output layer is column-parallel, so the logits are [b s v:tp]. Gathering them would move
b*s*v numbers, and v is huge. Compute the loss without that, and keep it numerically stable
by subtracting the global max. How many bytes do you move compared with gathering the logits?

Ops: amax(t, dim, keepdim) over a split dim gives a partial MAX (reduce with all_reduce(..., op="max"));
sum, exp, log; take(t, ids, dim) picks one entry per row (a partial sum if that dim is split).

Read: Megatron-LM `vocab_parallel_cross_entropy`; PyTorch `loss_parallel`.
"""

import torch.nn.functional as F

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class VocabParallelCrossEntropy(Module):
    def on_init(self, p):
        self.W = shard(p.W, "d v:tp")

    def forward(self, x, targets):
        x, targets = x.expect("b s d"), targets.expect("b s")
        logits = (x @ self.W).expect("b s v:tp")
        # TODO: the global max per token
        m = TODO
        # TODO: the global sum of exp(logits - m), and the target logit
        sumexp = TODO
        target = TODO
        self.x, self.targets, self.logits, self.m, self.sumexp = x, targets, logits, m, sumexp  # saved for 06b
        return (log(sumexp) + m).view("b s") - target


# ---- puzzle definition -------------------------------------------------------
def reference(c, p, x, targets):
    logits = x @ p.W
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view(targets.shape)


SPEC = Spec(
    dims="b s d v",
    params={"W": "d v"},
    inputs={"x": "b s d", "targets": Ids("b s", high="v")},
    output="b s",
    configs=[
        {"b": 2, "s": 4, "d": 6, "v": 12, "tp": 2},
        {"b": 1, "s": 5, "d": 8, "v": 16, "tp": 4},
        {"b": 2, "s": 3, "d": 4, "v": 9, "tp": 3},
    ],
    reference=reference,
    max_weight_elements='d*v/tp',
    max_forward_collectives=3,
    max_comm_bytes="12*b*s*(tp-1)/tp",
)
