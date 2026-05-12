"""
Comprehensive test suite for verification pipeline based on mathematical ground truth.
Following the verification plan requirements for behavior-focused testing.
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from src.models.manager import ModelManager
from src.pipeline.reasoning.types import ReasoningOutput
from src.pipeline.verification.verification import VerificationPipeline, VerificationResult
from src.pipeline.verification.verification_types import ErrorType


class MathematicalTestCase:
    """
    Structure test cases around mathematical truth, not code behavior.
    Following the verification plan pattern.
    """
    def __init__(self, problem: str, correct_answer: str, wrong_answers: list = None):
        self.problem = problem
        self.correct_answer = correct_answer
        self.wrong_answers = wrong_answers or []
        self.correct_solution_approach = None
        self.common_mistakes = []

    def create_correct_reasoning(self) -> ReasoningOutput:
        """Generate reasoning that should verify as correct"""
        return ReasoningOutput(
            original_problem=self.problem,
            worked_solution=self.correct_solution_approach or f"Solution steps for {self.problem}",
            final_answer=self.correct_answer,
            think_reasoning="<think>Mathematical reasoning</think>",
            processing_metadata={"test_case": "correct"}
        )

    def create_flawed_reasoning(self, flaw_type: str) -> ReasoningOutput:
        """Generate reasoning with specific mathematical errors"""
        wrong_answer = self.wrong_answers[0] if self.wrong_answers else "wrong_answer"
        return ReasoningOutput(
            original_problem=self.problem,
            worked_solution=f"Flawed solution with {flaw_type}",
            final_answer=wrong_answer,
            think_reasoning="<think>Flawed reasoning</think>",
            processing_metadata={"test_case": "flawed", "flaw_type": flaw_type}
        )


# Mathematical test cases covering different categories
ALGEBRAIC_CASES = [
    MathematicalTestCase(
        problem="Solve for x: 2x + 5 = 13",
        correct_answer="x = 4",
        wrong_answers=["x = 5", "x = 3"]
    ),
    MathematicalTestCase(
        problem="Factor: x² - 5x + 6",
        correct_answer="(x - 2)(x - 3)",
        wrong_answers=["(x - 1)(x - 6)", "(x + 2)(x + 3)"]
    ),
    MathematicalTestCase(
        problem="Solve the quadratic: x² - 4x - 5 = 0",
        correct_answer="x = 5 or x = -1",
        wrong_answers=["x = 4 or x = 1", "x = 5 or x = 1"]
    )
]

ARITHMETIC_CASES = [
    MathematicalTestCase(
        problem="Calculate: 15 × 24",
        correct_answer="360",
        wrong_answers=["350", "370", "340"]
    ),
    MathematicalTestCase(
        problem="What is 2³ + 3²?",
        correct_answer="17",
        wrong_answers=["16", "18", "15"]
    )
]

CALCULUS_CASES = [
    MathematicalTestCase(
        problem="Find the derivative of x³ + 2x",
        correct_answer="3x² + 2",
        wrong_answers=["3x² + 2x", "x² + 2", "3x³ + 2x"]
    )
]


class TestVerificationPipeline:
    """Main test suite for verification pipeline"""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager with realistic responses"""
        manager = Mock(spec=ModelManager)

        # Mock config structure matching your system
        manager.config = {
            "tasks": {
                "verification": {
                    "model": "qwen2.5-coder:7b-instruct",
                    "params": {"temperature": 0.1, "max_tokens": 5000},
                    "prompt_ref": "codegen/baseline_codegen@v7",
                    "provider": "ollama_local",
                    "max_repair_attempts": 3,
                    "confidence_threshold": 0.95,
                    "min_acceptable_confidence": 0.7,
                    "repair_temperature": 0.2,
                    "execution_timeout": 10,
                    "memory_limit_mb": 512
                }
            }
        }

        return manager

    @pytest.fixture
    def pipeline_with_mock_llm(self, mock_model_manager):
        """
        Create pipeline with controlled LLM responses.
        IMPORTANT: Mock simulates realistic LLM behavior, not perfect behavior.
        """
        def mock_llm_response(*args, **kwargs):
            # Check the reasoning object to determine response
            variables = kwargs.get('variables', {})
            reasoning = variables.get('reasoning')

            if reasoning and hasattr(reasoning, 'original_problem'):
                problem = reasoning.original_problem.lower()

                if "2x + 5 = 13" in problem:
                    return self._get_linear_equation_code(reasoning)
                elif "x² - 5x + 6" in problem:
                    return self._get_factoring_code(reasoning)
                elif "15 × 24" in problem:
                    return self._get_arithmetic_code(reasoning)
                elif "derivative" in problem:
                    return self._get_derivative_code(reasoning)

            # Default response for unknown problems
            return self._get_generic_verification_code()

        # Mock the call method
        mock_response = Mock()
        mock_response.content = ""
        mock_model_manager.call.side_effect = lambda *args, **kwargs: (
            setattr(mock_response, 'content', mock_llm_response(*args, **kwargs)) or mock_response
        )

        return VerificationPipeline(mock_model_manager)

    def _emit_helpers(self):
        return """
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

    def _get_linear_equation_code(self, reasoning=None):
        """Realistic SymPy code for linear equation"""
        claimed_answer = reasoning.final_answer if reasoning else "x = 4"
        return f"""```python
import sympy as sp
import json
{self._emit_helpers()}

# Problem Setup
x = sp.Symbol('x')
equation = sp.Eq(2*x + 5, 13)

# Step Verification
emit_step(1, "Setting up equation 2x + 5 = 13", True, str(equation))

# Answer Computation
solution = sp.solve(equation, x)
computed_answer = f"x = {{solution[0]}}"

# Answer Verification
claimed_answer = {claimed_answer!r}
answer_matches = claimed_answer == "x = 4"
emit_final(answer_matches, claimed_answer, f"Computed: {{computed_answer}}")
```"""

    def _get_arithmetic_code(self, reasoning=None):
        """Realistic SymPy code for arithmetic"""
        claimed_answer = reasoning.final_answer if reasoning else "360"
        return f"""```python
import sympy as sp
import json
{self._emit_helpers()}

# Problem Setup
# Calculate 15 × 24

# Step Verification
result = 15 * 24
emit_step(1, "Computing 15 × 24", result == 360, f"result={{result}}")

# Answer Computation
computed_answer = result

# Answer Verification
claimed_answer = {claimed_answer!r}
answer_matches = str(computed_answer) == claimed_answer
emit_final(answer_matches, claimed_answer, f"Computed: {{computed_answer}}")
```"""

    def _get_factoring_code(self, reasoning=None):
        """Realistic SymPy code for factoring"""
        claimed_answer = reasoning.final_answer if reasoning else "(x - 2)(x - 3)"
        return f"""```python
import sympy as sp
import json
{self._emit_helpers()}

# Problem Setup
x = sp.Symbol('x')
expr = x**2 - 5*x + 6

# Step Verification
factored = sp.factor(expr)
emit_step(1, "Factoring x² - 5x + 6", True, str(factored))

# Answer Computation
computed_answer = str(factored)

# Answer Verification
claimed_answer = {claimed_answer!r}
# Check if factorizations are equivalent
expanded_computed = sp.expand(factored)
expanded_claimed = sp.expand((x - 2)*(x - 3))
answer_matches = expanded_computed == expanded_claimed
emit_final(answer_matches, claimed_answer, f"Computed: {{computed_answer}}")
```"""

    def _get_derivative_code(self, reasoning=None):
        """Realistic SymPy code for derivatives"""
        claimed_answer = reasoning.final_answer if reasoning else "3x² + 2"
        return f"""```python
import sympy as sp
import json
{self._emit_helpers()}

# Problem Setup
x = sp.Symbol('x')
expr = x**3 + 2*x

# Step Verification
derivative = sp.diff(expr, x)
emit_step(1, "Finding derivative of x³ + 2x", True, str(derivative))

# Answer Computation
computed_answer = str(derivative)

# Answer Verification
claimed_answer = {claimed_answer!r}
answer_matches = claimed_answer in {{"3*x**2 + 2", "3x² + 2"}}
emit_final(answer_matches, claimed_answer, f"Computed: {{computed_answer}}")
```"""

    def _get_generic_verification_code(self):
        """Generic verification code for unknown problems"""
        return f"""```python
import sympy as sp
import json
{self._emit_helpers()}

emit_step(1, "Generic verification", True, "generic")
emit_final(True, "generic", "generic")
```"""

    def test_distinguishes_correct_from_incorrect(self, pipeline_with_mock_llm):
        """
        The most fundamental test: can it tell right from wrong?
        This MUST work or the system is useless.
        """
        correct_case = ARITHMETIC_CASES[0]

        correct_reasoning = correct_case.create_correct_reasoning()
        incorrect_reasoning = correct_case.create_flawed_reasoning("arithmetic_error")

        correct_result = pipeline_with_mock_llm.verify(correct_reasoning)
        incorrect_result = pipeline_with_mock_llm.verify(incorrect_reasoning)

        # This MUST work or the system is useless
        assert correct_result.confidence_score > incorrect_result.confidence_score
        # Note: We can't test answer_match without actual execution

    def test_handles_different_math_categories(self, pipeline_with_mock_llm):
        """Test that verification works across mathematical domains"""
        test_cases = [
            ALGEBRAIC_CASES[0],  # Linear equation
            ARITHMETIC_CASES[0],  # Basic arithmetic
            CALCULUS_CASES[0]     # Derivative
        ]

        results = []
        for case in test_cases:
            reasoning = case.create_correct_reasoning()
            result = pipeline_with_mock_llm.verify(reasoning)
            results.append(result)

        # All should complete without errors
        for result in results:
            assert result.status in ["verified", "failed_reasoning"]
            assert result.confidence_score >= 0.0

    def test_pipeline_error_handling(self, pipeline_with_mock_llm):
        """Test that pipeline handles various error conditions gracefully"""
        # Test with malformed reasoning
        malformed_reasoning = ReasoningOutput(
            original_problem="",  # Empty problem
            worked_solution="",   # Empty solution
            final_answer="",      # Empty answer
            think_reasoning="",
            processing_metadata={}
        )

        result = pipeline_with_mock_llm.verify(malformed_reasoning)

        # Should handle gracefully, not crash
        assert result is not None
        assert hasattr(result, 'status')
        assert hasattr(result, 'confidence_score')

    def test_repair_mechanism_structure(self, pipeline_with_mock_llm):
        """
        Test that repair mechanism has proper structure.
        We can't test actual repair without real execution, but we can test the flow.
        """
        reasoning = ALGEBRAIC_CASES[0].create_correct_reasoning()

        repair_prompt = pipeline_with_mock_llm._create_codegen_repair_prompt(
            "test code", "Test error"
        )

        assert isinstance(repair_prompt, str)
        assert len(repair_prompt) > 0
        assert "emit_step" in repair_prompt
        assert "emit_final" in repair_prompt


class TestMathematicalValidation:
    """Test mathematical correctness validation"""

    def test_arithmetic_validation(self):
        """Test basic arithmetic validation logic"""
        # Test the mathematical test cases themselves
        case = ARITHMETIC_CASES[0]  # 15 × 24 = 360

        assert case.problem == "Calculate: 15 × 24"
        assert case.correct_answer == "360"
        assert "350" in case.wrong_answers

        correct = case.create_correct_reasoning()
        incorrect = case.create_flawed_reasoning("arithmetic_error")

        assert correct.final_answer == "360"
        assert incorrect.final_answer != "360"

    def test_algebraic_validation(self):
        """Test algebraic problem validation"""
        case = ALGEBRAIC_CASES[0]  # 2x + 5 = 13

        correct = case.create_correct_reasoning()
        assert "x = 4" in correct.final_answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
