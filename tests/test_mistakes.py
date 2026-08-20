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
