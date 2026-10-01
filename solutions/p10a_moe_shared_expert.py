from shardpuzzles import *


class SharedExpertMoE(Module):
    def on_init(self, p):
        self.Wr, self.bias = p.Wr, p.bias
        self.W1 = shard(p.W1, "E:ep d f")
        self.W2 = shard(p.W2, "E:ep f d")
        # TODO: the shared expert's weights
        self.Ws1, self.Ws2 = p.Ws1, p.Ws2  # replicated: it runs on each rank's own tokens
        # END TODO

    def forward(self, x):
        x = x.expect("T:ep d")
        # TODO: sigmoid scores, selection with the bias, weights without it
        s = sigmoid((x @ self.Wr).expect("T:ep E"))
        _, expert_ids = topk(s + self.bias, "k", dim=-1)
        g = gather(s, expert_ids, dim=-1)
        g_sum = sum(g, -1, keepdim=True).expect("T:ep 1")
        weights = g / g_sum
        # END TODO
        buffers, plan = dispatch(x, expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep d"), "C", "E", ep).expect(
            "E:ep C d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        # TODO: the shared expert (no dependency on the all_to_all, so it could overlap with it)
        hs = (x @ self.Ws1).expect("T:ep fs")
        shared = (gelu(hs) @ self.Ws2).expect("T:ep d")
        # END TODO
        h = (buffers @ self.W1).expect("E:ep C f")
        y = all_to_all((gelu(h) @ self.W2).expect("E:ep C d"), "E", "C", ep).expect(
            "E C:ep d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        y = combine(y, plan).expect("T:ep k d")
        self.x, self.s, self.expert_ids, self.g_sum, self.weights = x, s, expert_ids, g_sum, weights  # for 10b
        self.plan, self.buffers, self.h, self.y, self.hs = plan, buffers, h, y, hs
        return sum(y * weights.view("T:ep k 1"), "k").expect("T:ep d") + shared


HINTS = [
    "s = sigmoid(logits); choose ids with topk(s + bias). The weights are gather(s, ids) divided by their sum, with no bias.",
    "The shared expert is a plain MLP on local tokens with replicated weights: no communication. Add it to the routed output.",
]
