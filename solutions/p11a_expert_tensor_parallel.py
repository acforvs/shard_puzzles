from shardpuzzles import *


class ExpertTensorParallelMoE(Module):
    def on_init(self, p):
        self.Wr = p.Wr
        # TODO: experts split over ep, and each expert split Megatron-style over tp
        self.W1 = shard(p.W1, "E:ep d f:tp")
        self.W2 = shard(p.W2, "E:ep f:tp d")
        # END TODO

    def forward(self, x):
        x = x.expect("T:ep d")
        top, expert_ids = topk((x @ self.Wr).expect("T:ep E"), "k", dim=-1)
        weights = softmax(top, dim=-1)
        buffers, plan = dispatch(x, expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep d"), "C", "E", ep).expect(
            "E:ep C d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        # TODO: the tp-split experts, then the return trip with d/tp-wide vectors
        h = (buffers @ self.W1).expect("E:ep C f:tp")
        y = (gelu(h) @ self.W2).expect("E:ep C d", partial=tp)
        y = reduce_scatter(y, "d", tp).expect("E:ep C d:tp", bytes="2*E/ep*C*d * (tp-1)/tp")
        y = all_to_all(y, "E", "C", ep).expect("E C:ep d:tp", bytes="2*E*C/ep*d/tp * (ep-1)/ep")
        # END TODO
        # TODO: combine on d:tp, gather d at the very end
        y = combine(y, plan).expect("T:ep k d:tp")
        out = sum(y * weights.view("T:ep k 1"), "k").expect("T:ep d:tp")
        out = all_gather(out, "d", tp).expect("T:ep d", bytes="2*T/ep*d * (tp-1)/tp")
        # END TODO
        self.x, self.expert_ids, self.weights, self.plan = x, expert_ids, weights, plan  # saved for 11b
        self.buffers, self.h, self.y = buffers, h, y
        return out


HINTS = [
    "Experts: 'E:ep d f:tp' and 'E:ep f:tp d'. After @ W2 the output is partial over tp.",
    "reduce_scatter along d over tp gives [E:ep C d:tp], so the return all_to_all is tp times smaller. combine and the weighted sum work on d:tp; all_gather over tp at the very end.",
]
