from shardpuzzles import *


class VocabParallelCrossEntropy(Module):
    def on_init(self, p):
        self.W = shard(p.W, "d v:tp")

    def forward(self, x, targets):
        x, targets = x.expect("b s d"), targets.expect("b s")
        logits = (x @ self.W).expect("b s v:tp")
        # TODO: the global max per token
        m = amax(logits, -1, keepdim=True).expect("b s 1", partial_max=tp)
        m = all_reduce(m, tp, op="max").expect("b s 1")
        # END TODO
        # TODO: the global sum of exp(logits - m), and the target logit
        sumexp = sum(exp(logits - m), -1, keepdim=True).expect("b s 1", partial=tp)
        sumexp = all_reduce(sumexp, tp).expect("b s 1")
        target = take(logits, targets, -1).expect("b s", partial=tp)  # only the owning rank contributes
        target = all_reduce(target, tp).expect("b s")
        # END TODO
        self.x, self.targets, self.logits, self.m, self.sumexp = x, targets, logits, m, sumexp  # saved for 06b
        return (log(sumexp) + m).view("b s") - target


HINTS = [
    "Never gather the logits. Reduce three per-token numbers instead: the max, the sum of exps and the target logit.",
    "amax over v:tp is a partial max -> all_reduce(op='max'); sum(exp(logits - m)) -> all_reduce; take(logits, targets) -> all_reduce.",
]
