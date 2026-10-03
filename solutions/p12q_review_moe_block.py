from shardpuzzles import *


class MoEBlock(Module):
    def on_init(self, p):
        self.Wr, self.bias, self.Wdown, self.Wup, self.Ws1, self.Ws2 = p.Wr, p.bias, p.Wdown, p.Wup, p.Ws1, p.Ws2
        self.W1 = shard(p.W1, "E:ep l f")
        self.W2 = shard(p.W2, "E:ep f l")

    def forward(self, x):
        x = x.expect("T:ep d")
        s = sigmoid((x @ self.Wr).expect("T:ep E"))
        _, expert_ids = topk(s + self.bias, "k", dim=-1)
        g = gather(s, expert_ids, dim=-1)
        weights = g / sum(g, -1, keepdim=True).expect("T:ep 1")

        buffers, plan = dispatch((x @ self.Wdown).expect("T:ep l"), expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep l"), "C", "E", ep).expect(
            "E:ep C l", bytes="2*E*C/ep*l * (ep-1)/ep"
        )
        y = (gelu((buffers @ self.W1).expect("E:ep C f")) @ self.W2).expect("E:ep C l")
        y = all_to_all(y, "E", "C", ep).expect("E C:ep l", bytes="2*E*C/ep*l * (ep-1)/ep")
        routed = (
            sum(combine(y, plan).expect("T:ep k l") * weights.view("T:ep k 1"), "k").expect("T:ep l") @ self.Wup
        ).expect("T:ep d")

        shared = (gelu((x @ self.Ws1).expect("T:ep fs")) @ self.Ws2).expect("T:ep d")
        return routed + shared


HINTS = ["Everything before dispatch and after combine is on this rank's tokens; only the [E C l] buffers travel."]
