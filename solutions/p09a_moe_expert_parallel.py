from shardpuzzles import *


class ExpertParallelMoE(Module):
    def on_init(self, p):
        self.Wr = p.Wr  # the router is small and replicated
        # TODO: give each rank E/ep experts
        self.W1 = shard(p.W1, "E:ep d f")
        self.W2 = shard(p.W2, "E:ep f d")
        # END TODO

    def forward(self, x):
        x = x.expect("T:ep d")  # this rank's tokens
        top, expert_ids = topk((x @ self.Wr).expect("T:ep E"), "k", dim=-1)
        weights = softmax(top, dim=-1)
        buffers, plan = dispatch(x, expert_ids, "E", "C")
        # TODO: send every buffer slot to the rank that owns its expert, run the experts, send results back
        buffers = all_to_all(buffers.expect("E C:ep d"), "C", "E", ep).expect(
            "E:ep C d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        h = (buffers @ self.W1).expect("E:ep C f")
        y = (gelu(h) @ self.W2).expect("E:ep C d")
        y = all_to_all(y, "E", "C", ep).expect("E C:ep d", bytes="2*E*C/ep*d * (ep-1)/ep")
        # END TODO
        # TODO: unpack the choices and mix them with the router weights
        y = combine(y, plan).expect("T:ep k d")
        out = sum(y * weights.view("T:ep k 1"), "k").expect("T:ep d")
        # END TODO
        self.x, self.expert_ids, self.weights, self.plan = x, expert_ids, weights, plan  # saved for 09b
        self.buffers, self.h, self.y = buffers, h, y
        return out


HINTS = [
    "dispatch gives [E C:ep d]: your slice of every expert's queue. all_to_all(gather_dim='C', scatter_dim='E') gives [E:ep C d]: every slot of your own experts.",
    "The experts are a batched matmul over E:ep. Then all_to_all back (gather 'E', scatter 'C'), combine, weight by the router weights, sum over k.",
    "Bytes per all_to_all: each rank holds E*(C/ep)*d of buffer and (ep-1)/ep of it leaves.",
]
