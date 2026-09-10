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
