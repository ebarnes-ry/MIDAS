from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.pipeline.reasoning.types import ReasoningOutput, ReasoningStep
from src.pipeline.verification.verification import VerificationPipeline


HELPERS = """
import sympy as sp
import json
import math
import itertools

def same_expr(a, b):
    try:
        return bool(sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0)
    except Exception:
        return False

def same_equation(a, b):
    try:
        a_lhs, a_rhs = a.lhs, a.rhs
        b_lhs, b_rhs = b.lhs, b.rhs
        same_order = same_expr(a_lhs, b_lhs) and same_expr(a_rhs, b_rhs)
        swapped_order = same_expr(a_lhs, b_rhs) and same_expr(a_rhs, b_lhs)
        return bool(same_order or swapped_order)
    except Exception:
        return False

def to_json_bool(value):
    if value is True:
        return True
    if value is False:
        return False
    if value == sp.S.true:
        return True
    if value == sp.S.false:
        return False
    try:
        return bool(value)
    except Exception:
        return False

def emit_step(step, description, verified, note=""):
    print(json.dumps({
        "step": int(step),
        "description": str(description),
        "verified": to_json_bool(verified),
        "note": str(note),
    }))

def emit_final(verified, answer, note=""):
    print(json.dumps({
        "final_answer_verified": to_json_bool(verified),
        "answer": str(answer),
        "note": str(note),
    }))
"""


def _manager():
    return SimpleNamespace(
        config={
            "tasks": {
                "verification": {
                    "prompt_ref": "codegen/baseline_codegen@v7",
                    "repair_temperature": 0.1,
                    "execution_timeout": 10,
                    "memory_limit_mb": 512,
                }
            }
        }
    )


def _reasoning(problem: str, claim: str, latex: str, answer: str) -> ReasoningOutput:
    return ReasoningOutput(
        original_problem=problem,
        steps=[
            ReasoningStep(
                step_number=1,
                claim=claim,
                latex_expression=latex,
                justification="Verify the symbolic result directly.",
            )
        ],
        final_answer=answer,
        think_reasoning="",
    )


CASES = [
    (
        "linear equation",
        _reasoning("Solve 3*x + 2 = 11.", "Solving gives x = 3.", "x = 3", "x = 3"),
        """
x = sp.Symbol("x")
solution = sp.solve(sp.Eq(3*x + 2, 11), x)[0]
emit_step(1, "Solving gives x = 3.", solution == 3, f"computed x={solution}")
emit_final(solution == 3, "x = 3", f"computed=x = {solution}; claimed=x = 3")
""",
    ),
    (
        "linear system",
        _reasoning(
            "Solve x + y = 5 and x - y = 1.",
            "The solution is x = 3 and y = 2.",
            "x = 3, y = 2",
            "x = 3, y = 2",
        ),
        """
x, y = sp.symbols("x y")
solution = sp.solve([sp.Eq(x + y, 5), sp.Eq(x - y, 1)], [x, y])
verified = solution[x] == 3 and solution[y] == 2
emit_step(1, "The solution is x = 3 and y = 2.", verified, f"computed={solution}")
emit_final(verified, "x = 3, y = 2", f"computed=x = {solution[x]}, y = {solution[y]}; claimed=x = 3, y = 2")
""",
    ),
    (
        "factoring",
        _reasoning(
            "Factor x^2 - 5*x + 6.",
            "The factorization is (x - 2)(x - 3).",
            "(x - 2)(x - 3)",
            "(x - 2)(x - 3)",
        ),
        """
x = sp.Symbol("x")
factored = sp.factor(x**2 - 5*x + 6)
verified = same_expr(factored, (x - 2)*(x - 3))
emit_step(1, "The factorization is (x - 2)(x - 3).", verified, f"computed={factored}")
emit_final(verified, "(x - 2)(x - 3)", f"computed={factored}; claimed=(x - 2)(x - 3)")
""",
    ),
    (
        "expansion",
        _reasoning(
            "Expand (x + 2)(x - 3).",
            "The expansion is x^2 - x - 6.",
            "x^2 - x - 6",
            "x^2 - x - 6",
        ),
        """
x = sp.Symbol("x")
expanded = sp.expand((x + 2)*(x - 3))
verified = same_expr(expanded, x**2 - x - 6)
emit_step(1, "The expansion is x^2 - x - 6.", verified, f"computed={expanded}")
emit_final(verified, "x^2 - x - 6", f"computed={expanded}; claimed=x^2 - x - 6")
""",
    ),
    (
        "derivative",
        _reasoning(
            "Differentiate x^3 + 2*x.",
            "The derivative is 3*x^2 + 2.",
            "3*x^2 + 2",
            "3*x^2 + 2",
        ),
        """
x = sp.Symbol("x")
derivative = sp.diff(x**3 + 2*x, x)
verified = same_expr(derivative, 3*x**2 + 2)
emit_step(1, "The derivative is 3*x^2 + 2.", verified, f"computed={derivative}")
emit_final(verified, "3*x^2 + 2", f"computed={derivative}; claimed=3*x^2 + 2")
""",
    ),
    (
        "simple integral",
        _reasoning(
            "Integrate 2*x from 0 to 3.",
            "The definite integral is 9.",
            "\\int_0^3 2x dx = 9",
            "9",
        ),
        """
x = sp.Symbol("x")
integral = sp.integrate(2*x, (x, 0, 3))
verified = integral == 9
emit_step(1, "The definite integral is 9.", verified, f"computed={integral}")
emit_final(verified, "9", f"computed={integral}; claimed=9")
""",
    ),
    (
        "limit",
        _reasoning(
            "Find the limit of sin(x)/x as x approaches 0.",
            "The limit is 1.",
            "\\lim_{x\\to 0} \\sin(x)/x = 1",
            "1",
        ),
        """
x = sp.Symbol("x")
limit_value = sp.limit(sp.sin(x)/x, x, 0)
verified = limit_value == 1
emit_step(1, "The limit is 1.", verified, f"computed={limit_value}")
emit_final(verified, "1", f"computed={limit_value}; claimed=1")
""",
    ),
    (
        "combinatorics count",
        _reasoning(
            "How many ways are there to choose 2 items from 5?",
            "There are 10 choices.",
            "\\binom{5}{2} = 10",
            "10",
        ),
        """
count = math.comb(5, 2)
verified = count == 10
emit_step(1, "There are 10 choices.", verified, f"computed={count}")
emit_final(verified, "10", f"computed={count}; claimed=10")
""",
    ),
    (
        "modular arithmetic",
        _reasoning(
            "Find 17 mod 5.",
            "17 is congruent to 2 modulo 5.",
            "17 \\equiv 2 \\pmod 5",
            "2",
        ),
        """
remainder = 17 % 5
verified = remainder == 2
emit_step(1, "17 is congruent to 2 modulo 5.", verified, f"computed={remainder}")
emit_final(verified, "2", f"computed={remainder}; claimed=2")
""",
    ),
    (
        "matrix computation",
        _reasoning(
            "Multiply [[1, 2], [3, 4]] by [[1], [1]].",
            "The product is [[3], [7]].",
            "\\begin{bmatrix}3\\\\7\\end{bmatrix}",
            "[[3], [7]]",
        ),
        """
product = sp.Matrix([[1, 2], [3, 4]]) * sp.Matrix([[1], [1]])
expected = sp.Matrix([[3], [7]])
verified = product == expected
emit_step(1, "The product is [[3], [7]].", verified, f"computed={product}")
emit_final(verified, "[[3], [7]]", f"computed={product.tolist()}; claimed=[[3], [7]]")
""",
    ),
]


@pytest.mark.parametrize("case_name,reasoning,body", CASES, ids=[case[0] for case in CASES])
def test_core_math_verification_regressions(case_name, reasoning, body):
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock(return_value=(HELPERS + body, {"case": case_name}))

    result = pipeline.verify(reasoning)

    assert result.status == "verified"
    assert result.answer_match is True
    assert result.step_verifications[0].verified is True
    assert result.metadata["final_verdict"]["final_answer_verified"] is True


def test_missing_expected_step_output_is_failed_contract():
    reasoning = ReasoningOutput(
        original_problem="Check two steps.",
        steps=[
            ReasoningStep(1, "First step is valid.", "Given.", "x = x"),
            ReasoningStep(2, "Second step is valid.", "Given.", "y = y"),
        ],
        final_answer="ok",
        think_reasoning="",
    )
    code = HELPERS + """
emit_step(1, "First step is valid.", True, "confirmed")
emit_final(True, "ok", "computed=ok; claimed=ok")
"""
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock(return_value=(code, {}))
    pipeline._get_repaired_code = Mock(side_effect=ValueError("No repaired code"))

    result = pipeline.verify(reasoning)

    assert result.status == "failed_contract"
    assert "missing step output(s): 2" in result.errors[0].message
