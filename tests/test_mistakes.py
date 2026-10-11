"""Common mistakes fail with useful diagnoses and no computed byte answers."""

import inspect

import pytest

from shardpuzzles.catalog import discover
from shardpuzzles.errors import PuzzleError, WrongResult
from shardpuzzles.runner import run

PUZZLES = discover()


def patched_solution(pid, old, new):
    cls = PUZZLES[pid].module_class(solution=True)
    path = inspect.getsourcefile(cls)
    assert path is not None
    source = open(path).read()
    assert old in source
    namespace = {}
    exec(compile(source.replace(old, new, 1), path, "exec"), namespace)
    return namespace[cls.__name__]


def check(pid, cls):
    spec = PUZZLES[pid].spec
    return run(spec, cls, spec.configs[0])


def test_forgetting_the_weight_gradient_all_reduce():
    wrong = patched_solution("00c", '        dW = all_reduce(dW, dp).expect("d f", bytes="4*d*f * (dp-1)/dp")\n', "")
    with pytest.raises(WrongResult, match=r"grad 'W' is .*partial\(dp\)"):
        check("00c", wrong)


def test_forgetting_the_tensor_parallel_input_gradient_all_reduce():
    wrong = patched_solution("01b", '        dx = all_reduce(dx, tp).expect("b s d")\n', "")
    with pytest.raises(WrongResult, match=r"grad 'x' is .*partial\(tp\)"):
        check("01b", wrong)


def test_forgetting_the_sequence_parallel_input_all_gather():
    wrong = patched_solution(
        "02a", '        x = all_gather(x, "s", tp).expect("b s d", bytes="2*b*s*d * (tp-1)/tp")\n', ""
    )
    with pytest.raises(PuzzleError, match="cannot split two dimensions"):
        check("02a", wrong)


def test_forgetting_the_sequence_parallel_input_gradient_reduce_scatter():
    wrong = patched_solution("02b", '        dx = reduce_scatter(dx, "s", tp).expect("b s:tp d")\n', "")
    with pytest.raises(WrongResult, match=r"grad 'x' is .*partial\(tp\)"):
        check("02b", wrong)


def test_attention_softmax_must_normalize_keys():
    wrong = patched_solution("03a", 'causal_mask("s s"), dim=-1)', 'causal_mask("s s"), dim=-2)')
    with pytest.raises(WrongResult, match="right layout but the wrong numbers"):
        check("03a", wrong)


def test_shared_kv_head_gradients_are_summed_instead_of_averaged():
    wrong = patched_solution(
        "03b",
        'dk = repeat_interleave_backward(dk, "h/hkv", dim=1)',
        'dk = repeat_interleave_backward(dk, "h/hkv", dim=1) / size("h/hkv")',
    )
    with pytest.raises(WrongResult, match="right layout but the wrong numbers"):
        check("03b", wrong)


def test_naive_qkv_fusion_has_the_right_shapes_but_wrong_numbers():
    wrong = patched_solution(
        "04a",
        """        wq = p.Wq.view("d hkv (h/hkv)*hd")  # group the query columns by the kv head they use
        wk = p.Wk.view("d hkv hd")
        wv = p.Wv.view("d hkv hd")
        wqkv = shard(cat([wq, wk, wv], dim=2), "d hkv:tp (h/hkv+2)*hd")
        self.Wqkv = wqkv.view("d (hkv*(h/hkv+2)*hd):tp")""",
        """        self.Wqkv = shard(cat([p.Wq, p.Wk, p.Wv], dim=1), "d (h*hd+2*hkv*hd):tp")""",
    )
    with pytest.raises(WrongResult, match="right layout but the wrong numbers"):
        check("04a", wrong)


def test_unfused_einsum_projections_cannot_bypass_the_fusion_budget():
    unfused = patched_solution(
        "03a",
        """        q = (x @ self.Wq).expect("b s (h*hd):tp").view("b s h:tp hd").transpose(1, 2)
        k = (x @ self.Wk).expect("b s (hkv*hd):tp").view("b s hkv:tp hd").transpose(1, 2)
        v = (x @ self.Wv).expect("b s (hkv*hd):tp").view("b s hkv:tp hd").transpose(1, 2)""",
        """        q = einsum("b s d, d e -> b s e", x, self.Wq).expect("b s (h*hd):tp")
        q = q.view("b s h:tp hd").transpose(1, 2)
        k = einsum("b s d, d e -> b s e", x, self.Wk).expect("b s (hkv*hd):tp")
        k = k.view("b s hkv:tp hd").transpose(1, 2)
        v = einsum("b s d, d e -> b s e", x, self.Wv).expect("b s (hkv*hd):tp")
        v = v.view("b s hkv:tp hd").transpose(1, 2)""",
    )
    puzzle = PUZZLES["04a"]
    for config in puzzle.spec.configs:
        run(puzzle.spec.replace(max_weight_matmuls=None), unfused, config)
        with pytest.raises(WrongResult, match="forward multiplies by weights 4 times; at most 2"):
            run(puzzle.spec, unfused, config)


def test_embedding_backward_needs_all_tokens_gradient_rows():
    wrong = patched_solution("05b", '        dy = all_gather(dy, "s", tp).expect("b s d")\n', "")
    with pytest.raises(PuzzleError, match="must be laid out like the lookup's output"):
        check("05b", wrong)


def test_cross_entropy_maximum_needs_a_max_reduction():
    wrong = patched_solution("06a", 'all_reduce(m, tp, op="max")', "all_reduce(m, tp)")
    with pytest.raises(PuzzleError, match="use op='max'"):
        check("06a", wrong)


def test_cross_entropy_target_logit_needs_a_reduction():
    wrong = patched_solution("06a", '        target = all_reduce(target, tp).expect("b s")\n', "")
    with pytest.raises(PuzzleError, match="partial-ness differs"):
        check("06a", wrong)


def test_cross_entropy_input_gradient_needs_an_all_reduce():
    wrong = patched_solution("06b", '        dx = all_reduce(dx, tp).expect("b s d")\n', "")
    with pytest.raises(WrongResult, match=r"grad 'x' is .*partial\(tp\)"):
        check("06b", wrong)


@pytest.mark.parametrize("pid", ["07a", "08a", "07b", "08b"])
def test_fsdp_cannot_keep_gathered_weights_between_passes(pid):
    weight = "W2" if pid.endswith("b") else "W1"
    wrong = patched_solution(pid, f"{weight} = all_gather(", f"self.cached = {weight} = all_gather(")
    phase = "backward" if pid.endswith("b") else "forward"
    puzzle = PUZZLES[pid]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match=f"after {phase}, stored weights"):
            run(puzzle.spec, wrong, config)


@pytest.mark.parametrize(
    "pid,old,new,message",
    [
        (
            "07b",
            '        dW1 = reduce_scatter(dW1, "d", dp).expect("d:dp f", bytes="2*d*f * (dp-1)/dp")\n',
            '        dW1 = shard(dW1, "d:dp f")\n',
            "reduce_scatter it instead",
        ),
        (
            "08a",
            'all_gather(self.W1, "d", dp)',
            'all_gather(self.W1, "d", tp)',
            "isn't split over tp",
        ),
        (
            "08b",
            'reduce_scatter(dW2, "d", dp)',
            'reduce_scatter(dW2, "d", tp)',
            "isn't a partial sum over tp",
        ),
        (
            "08b",
            '        dx = all_reduce(dx, tp).expect("b:dp s d", bytes="4*b/dp*s*d * (tp-1)/tp")\n',
            "",
            r"grad 'x' is .*partial\(tp\)",
        ),
        (
            "08a",
            '"2*d*f/tp * (dp-1)/dp"',
            '"2*d*f * (dp-1)/dp"',
            r"incorrect byte count for all_gather\(dp\)",
        ),
    ],
)
def test_fsdp_requires_the_correct_reduction_group_and_volume(pid, old, new, message):
    wrong = patched_solution(pid, old, new)
    with pytest.raises(PuzzleError, match=message):
        check(pid, wrong)


@pytest.mark.parametrize(
    "old,new",
    [
        ("gather(s, expert_ids, dim=-1)", "gather(s + self.bias, expert_ids, dim=-1)"),
        ('topk(s + self.bias, "k", dim=-1)', 'topk(s, "k", dim=-1)'),
    ],
)
def test_shared_expert_bias_only_changes_selection(old, new):
    wrong = patched_solution("10a", old, new)
    with pytest.raises(WrongResult, match="right layout but the wrong numbers"):
        check("10a", wrong)


def test_moe_router_weight_gradient_needs_a_reduction():
    wrong = patched_solution(
        "09b",
        'all_reduce(einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep), ep).expect("d E")',
        'einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep)',
    )
    with pytest.raises(WrongResult, match=r"grad 'Wr' is .*partial\(ep\)"):
        check("09b", wrong)


def test_shared_expert_weight_gradient_needs_a_reduction():
    wrong = patched_solution(
        "10b",
        'all_reduce(einsum("t d, t f -> d f", self.x, dhs).expect("d fs", partial=ep), ep).expect("d fs")',
        'einsum("t d, t f -> d f", self.x, dhs).expect("d fs", partial=ep)',
    )
    with pytest.raises(WrongResult, match=r"grad 'Ws1' is .*partial\(ep\)"):
        check("10b", wrong)


def test_moe_combine_requires_the_return_all_to_all():
    wrong = patched_solution(
        "09a", '        y = all_to_all(y, "E", "C", ep).expect("E C:ep d", bytes="2*E*C/ep*d * (ep-1)/ep")\n', ""
    )
    with pytest.raises(PuzzleError, match="combine expects"):
        check("09a", wrong)


def test_expert_tensor_parallel_backward_slices_replicated_dy():
    wrong = patched_solution(
        "11b",
        'dy = shard(dy, "T:ep d:tp")',
        'dy = reduce_scatter(dy, "d", tp).expect("T:ep d:tp")',
    )
    with pytest.raises(PuzzleError, match="nothing to reduce"):
        check("11b", wrong)


def test_expert_tensor_parallel_router_choice_gradient_needs_a_tp_reduction():
    wrong = patched_solution("11b", '        dweights = all_reduce(dweights, tp).expect("T:ep k")\n', "")
    with pytest.raises(PuzzleError, match="partial sum over tp"):
        check("11b", wrong)


def test_expert_tensor_parallel_router_weight_gradient_reduces_over_ep():
    wrong = patched_solution("11b", 'partial=ep), ep).expect("d E")', 'partial=ep), tp).expect("d E")')
    with pytest.raises(PuzzleError, match="isn't a partial sum over tp"):
        check("11b", wrong)


def test_expert_tensor_parallel_return_bytes_use_the_local_hidden_width():
    wrong = patched_solution("11a", '"2*E*C/ep*d/tp * (ep-1)/ep"', '"2*E*C/ep*d * (ep-1)/ep"')
    with pytest.raises(PuzzleError, match=r"incorrect byte count for all_to_all\(ep\)"):
        check("11a", wrong)


@pytest.mark.parametrize(
    "gradient,contraction,layout",
    [("Wup", '"t l, t d -> l d", self.routed_l, dy', "l d"), ("Wdown", '"t d, t l -> d l", self.x, dz', "d l")],
)
def test_latent_projection_weight_gradients_need_a_reduction(gradient, contraction, layout):
    local = f'einsum({contraction}).expect("{layout}", partial=ep)'
    wrong = patched_solution("12b", f'all_reduce({local}, ep).expect("{layout}")', local)
    with pytest.raises(WrongResult, match=rf"grad '{gradient}' is .*partial\(ep\)"):
        check("12b", wrong)


def test_latent_moe_input_gradient_includes_the_router_path():
    wrong = patched_solution(
        "12b",
        "dz @ self.Wdown.transpose(0, 1) + dlogits @ self.Wr.transpose(0, 1)",
        "dz @ self.Wdown.transpose(0, 1)",
    )
    with pytest.raises(WrongResult, match="grad 'x'.*wrong numbers"):
        check("12b", wrong)


def test_latent_moe_dispatch_bytes_use_the_latent_width():
    wrong = patched_solution("12a", '"2*E*C/ep*l * (ep-1)/ep"', '"2*E*C/ep*d * (ep-1)/ep"')
    with pytest.raises(PuzzleError, match=r"incorrect byte count for all_to_all\(ep\)"):
        check("12a", wrong)


@pytest.mark.parametrize(
    "contraction,layout,gradient",
    [
        ('"b s e, b s d -> e d", self.o, dy', "h*hd d", "Wo"),
        ('"b s d, b s e -> d e", self.x, dq', "d h*hd", "Wq"),
    ],
)
def test_ulysses_replicated_weight_gradients_need_a_context_reduction(contraction, layout, gradient):
    local = f'einsum({contraction}).expect("{layout}", partial=cp)'
    wrong = patched_solution("13b", f'all_reduce({local}, cp).expect("{layout}")', local)
    with pytest.raises(WrongResult, match=rf"grad '{gradient}' is .*partial\(cp\)"):
        check("13b", wrong)


def test_ulysses_returns_the_attention_output_to_sequence_splits():
    wrong = patched_solution(
        "13a",
        '        o = all_to_all(o.transpose(1, 2), "h", "s", cp).expect('
        '"b s:cp h hd", bytes="2*b*s*h/cp*hd * (cp-1)/cp")\n',
        "        o = o.transpose(1, 2)\n",
    )
    with pytest.raises(PuzzleError, match="can't add, drop or move sharding"):
        check("13a", wrong)


def test_ulysses_unrepeats_kv_gradients_before_sending_them():
    late = patched_solution(
        "13b",
        """        dk = repeat_interleave_backward(dk, "h/hkv", dim=1)  # hkv heads are cheaper to send than h
        dv = repeat_interleave_backward(dv, "h/hkv", dim=1)
        dq = all_to_all(dq.transpose(1, 2), "h", "s", cp).expect("b s:cp h hd", bytes="2*b*s*h/cp*hd * (cp-1)/cp")
        dk = all_to_all(dk.transpose(1, 2), "hkv", "s", cp).expect("b s:cp hkv hd", bytes="2*b*s*hkv/cp*hd * (cp-1)/cp")
        dv = all_to_all(dv.transpose(1, 2), "hkv", "s", cp).expect("b s:cp hkv hd", bytes="2*b*s*hkv/cp*hd * (cp-1)/cp")
        """.rstrip(),
        """        dq = all_to_all(dq.transpose(1, 2), "h", "s", cp).expect("b s:cp h hd")
        dk = all_to_all(dk.transpose(1, 2), "h", "s", cp).expect("b s:cp h hd")
        dv = all_to_all(dv.transpose(1, 2), "h", "s", cp).expect("b s:cp h hd")
        dk = repeat_interleave_backward(dk, "h/hkv", dim=2)
        dv = repeat_interleave_backward(dv, "h/hkv", dim=2)""",
    )
    puzzle = PUZZLES["13b"]
    for config in puzzle.spec.configs:
        if config["cp"] == 1 or config["h"] == config["hkv"]:
            continue
        run(puzzle.spec.replace(max_comm_bytes=None), late, config)
        with pytest.raises(WrongResult, match="moved .*the budget is"):
            run(puzzle.spec, late, config)


def test_using_a_partial_result_as_a_replica():
    wrong = patched_solution("00b", '.expect("b s f", partial=tp)', '.expect("b s f")')
    with pytest.raises(PuzzleError, match="this is a partial sum over tp"):
        check("00b", wrong)


def test_bytes_are_required_where_the_exercise_asks():
    wrong = patched_solution("00b", ', bytes="4*b*s*f * (tp-1)/tp"', "")
    with pytest.raises(PuzzleError, match="also asks for the bytes"):
        check("00b", wrong)


def test_wrong_bytes_show_only_the_submitted_formula():
    wrong = patched_solution("00b", '"4*b*s*f * (tp-1)/tp"', '"2*b*s*f * (tp-1)/tp"')
    with pytest.raises(PuzzleError, match=r"incorrect byte count for all_reduce\(tp\)") as error:
        check("00b", wrong)
    message = str(error.value)
    assert "submitted: 2*b*f*s*(tp-1)/tp" in message
    assert "4*b*f*s*(tp-1)/tp" not in message
    assert "actual:" not in message


def test_equivalent_byte_formulas_are_accepted():
    alternative = patched_solution("00b", '"4*b*s*f * (tp-1)/tp"', '"2*2*b*s*f*(tp-1)/tp"')
    check("00b", alternative)


def test_returning_an_unchecked_collective():
    wrong = patched_solution(
        "00b",
        '        y = all_reduce(y, tp).expect("b s f", bytes="4*b*s*f * (tp-1)/tp")\n',
        "        y = all_reduce(y, tp)\n",
    )
    with pytest.raises(WrongResult, match="unchecked all_reduce"):
        check("00b", wrong)
