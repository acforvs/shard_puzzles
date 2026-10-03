"""Quiz generation preserves the operations and hides byte answers in scaffolds."""

import ast

import pytest

from shardpuzzles.catalog import discover
from shardpuzzles.errors import Unfinished
from shardpuzzles.runner import run
from shardpuzzles.scaffold import quiz, scaffold, solution_source, stub


class WithoutAnswers(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "expect":
            node.args, node.keywords = [], []
        return node


def test_quiz_preserves_operations_and_given_input_layouts():
    puzzle = discover()["12q"]
    generated = stub(puzzle)
    solution, _ = solution_source(puzzle)
    assert generated == quiz(puzzle)
    assert 'x = x.expect("T:ep d")' in generated
    assert '"???"' in generated
    normalized = [ast.dump(WithoutAnswers().visit(ast.parse(code))) for code in (generated, solution)]
    assert normalized[0] == normalized[1]


@pytest.mark.parametrize("generate", [stub, scaffold])
def test_quiz_and_scaffold_hide_every_byte_answer(generate):
    puzzle = discover()["12q"]
    code = generate(puzzle)
    assert "2*E*C/ep*l" not in code
    counts = [
        keyword.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "bytes"
    ]
    assert len(counts) == 2
    assert all(isinstance(value, ast.Constant) and value.value == "???" for value in counts)


def test_quiz_scaffold_stops_at_missing_bytes_and_accepts_completed_answers():
    puzzle = discover()["12q"]
    code = scaffold(puzzle)
    namespace = {}
    exec("from shardpuzzles import *\n" + code, namespace)
    with pytest.raises(Unfinished, match="fill in the"):
        run(puzzle.spec, namespace["MoEBlock"], puzzle.spec.configs[0])
    completed = code.replace('bytes="???"', 'bytes="2*E*C/ep*l*(ep-1)/ep"')
    exec("from shardpuzzles import *\n" + completed, namespace)
    for config in puzzle.spec.configs:
        run(puzzle.spec, namespace["MoEBlock"], config)
