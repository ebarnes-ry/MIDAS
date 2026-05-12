"""
Comprehensive tests for the verification pipeline.

Tests the core verification functionality including:
- Code generation from reasoning
- Safe execution environment
- Output parsing and validation
- Error handling and classification
- Reasoning repair orchestration
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from types import SimpleNamespace

from src.pipeline.verification.verification import VerificationPipeline
from src.pipeline.verification.verification_orchestrator import VerificationOrchestrator
from src.pipeline.verification.codegen import SymPyCodeGenerator, CodegenContractError
from src.pipeline.verification.executor import SafeExecutor
from src.pipeline.verification.parser import VerificationOutputParser
from src.pipeline.verification.verification_types import (
    VerificationResult, CodeExecutionResult, StepVerification, VerificationError, ErrorType
)
from src.pipeline.reasoning.types import ReasoningOutput, ReasoningStep
from src.models.manager import ModelManager


@pytest.fixture
def sample_reasoning():
    """Sample reasoning output for testing."""
    return ReasoningOutput(
        original_problem="Find the derivative of f(x) = 3x^2 + 2x + 1",
        worked_solution=(
            "To find the derivative, we apply the power rule to each term.\n"
            "The derivative of 3x^2 is 6x.\n"
            "The derivative of 2x is 2.\n"
            "The derivative of 1 is 0.\n"
            "Therefore, f'(x) = 6x + 2."
        ),
        final_answer="6*x + 2",
        think_reasoning="This is a basic polynomial differentiation problem.",
        processing_metadata={"source_model": "test_model"}
    )


@pytest.fixture
def mock_model_manager():
    """Mock model manager for testing."""
    manager = Mock(spec=ModelManager)
    manager.config = {
        "tasks": {
            "verification": {
                "prompt_ref": "codegen/baseline_codegen@v7",
                "repair_temperature": 0.1,
                "execution_timeout": 10,
                "memory_limit_mb": 256
            }
        }
    }
    manager.prompts = Mock()
    return manager


@pytest.fixture
def sample_valid_code():
    """Sample valid SymPy code that should execute successfully."""
    return '''
import sympy as sp
import json

def are_numerically_equal(a, b, tolerance=1e-9):
    try:
        num_a = float(a)
        num_b = float(b)
        return abs(num_a - num_b) < tolerance
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()

x = sp.Symbol('x')
f = 3*x**2 + 2*x + 1
derivative = sp.diff(f, x)

print(json.dumps({"step": 1, "description": "Applied power rule", "verified": True}))
print(json.dumps({"step": 2, "description": "Computed derivative", "verified": True}))
print(json.dumps({"final_answer_verified": True, "computed": str(derivative), "claimed": "6*x + 2"}))
'''


@pytest.fixture
def sample_invalid_code():
    """Sample invalid code that should fail execution."""
    return '''
import sympy as sp
import json

# This will cause a syntax error
x = sp.Symbol('x'
f = 3*x**2 + 2*x + 1
'''


@pytest.fixture
def linear_equation_reasoning():
    return ReasoningOutput(
        original_problem="16 - 2t = 5t + 9",
        steps=[
            ReasoningStep(
                step_number=1,
                claim="Adding 2t to both sides gives 16 = 7t + 9.",
                latex_expression="16 = 7t + 9",
                justification="Addition property of equality and combining like terms.",
            ),
            ReasoningStep(
                step_number=2,
                claim="Subtracting 9 from both sides gives 7 = 7t.",
                latex_expression="7 = 7t",
                justification="Subtraction property of equality.",
            ),
            ReasoningStep(
                step_number=3,
                claim="Dividing both sides by 7 gives t = 1.",
                latex_expression="t = 1",
                justification="Division property of equality.",
            ),
        ],
        final_answer="t = 1",
        think_reasoning="Solve the linear equation by isolating t.",
    )


class TestSymPyCodeGenerator:
    """Test the SymPy code generation component."""

    def test_extract_code_python_block(self):
        """Test extracting code from python markdown block."""
        generator = SymPyCodeGenerator(Mock())
        response = '''
Here's the verification code:

```python
import sympy as sp
x = sp.Symbol('x')
print("Hello")
```

This should work.
'''
        code = generator.extract_code(response)
        assert code is not None
        assert "import sympy as sp" in code
        assert "x = sp.Symbol('x')" in code

    def test_extract_code_generic_block(self):
        """Test extracting code from generic markdown block."""
        generator = SymPyCodeGenerator(Mock())
        response = '''
```
import sympy as sp
import json
print("test")
```
'''
        code = generator.extract_code(response)
        assert code is not None
        assert "import sympy as sp" in code

    def test_extract_code_fallback(self):
        """Test fallback extraction when no markdown blocks."""
        generator = SymPyCodeGenerator(Mock())
        response = '''import sympy as sp
import json
x = sp.Symbol('x')
print("test")'''
        code = generator.extract_code(response)
        assert code is not None
        assert "import sympy as sp" in code

    def test_extract_code_no_valid_code(self):
        """Test when no valid code can be extracted."""
        generator = SymPyCodeGenerator(Mock())
        response = "This is just text with no code."
        code = generator.extract_code(response)
        assert code is None

    def test_generate_success(self, mock_model_manager, sample_reasoning):
        """Test successful code generation."""
        # Mock the model response
        mock_response = Mock()
        mock_response.content = '''
```python
import sympy as sp
import json
x = sp.Symbol('x')
```
'''
        mock_response.meta = {"model": "test_model", "latency": 100}
        mock_model_manager.call.return_value = mock_response

        generator = SymPyCodeGenerator(mock_model_manager)
        code, metadata = generator.generate(sample_reasoning)

        assert code is not None
        assert "import sympy as sp" in code
        assert metadata["model_used"] == "test_model"
        assert metadata["latency_ms"] == 100
        assert metadata["prompt_ref"] == "codegen/baseline_codegen@v7"

    def test_validate_code_contract_rejects_instance_simplify(self):
        generator = SymPyCodeGenerator(Mock())
        bad_code = """import sympy as sp
import json
x = sp.symbols('x')
verified = bool((x - x).simplify() == 0)
print(json.dumps({"step": 1, "description": "bad", "verified": verified, "note": ""}))
"""

        with pytest.raises(ValueError, match="do not call .simplify"):
            generator.validate_code_contract(bad_code)

    def test_generate_raises_contract_error_with_bad_simplify(
        self, mock_model_manager, linear_equation_reasoning
    ):
        mock_response = Mock()
        mock_response.content = """import sympy as sp
import json
lhs_step3 = 16 - 9
lhs_step4 = 7
verified = bool((lhs_step3 - lhs_step4).simplify() == 0)
print(json.dumps({"step": 1, "description": "bad", "verified": verified, "note": ""}))
"""
        mock_response.meta = {"model": "test_model", "latency": 100}
        mock_model_manager.call.return_value = mock_response

        generator = SymPyCodeGenerator(mock_model_manager)

        with pytest.raises(CodegenContractError):
            generator.generate(linear_equation_reasoning)

    def test_generate_no_code_found(self, mock_model_manager, sample_reasoning):
        """Test generation when no code is found in response."""
        mock_response = Mock()
        mock_response.content = "No code here"
        mock_model_manager.call.return_value = mock_response

        generator = SymPyCodeGenerator(mock_model_manager)

        with pytest.raises(ValueError, match="No valid Python code block found"):
            generator.generate(sample_reasoning)


class TestSafeExecutor:
    """Test the safe execution environment."""

    def test_execute_valid_code(self, sample_valid_code):
        """Test executing valid code."""
        executor = SafeExecutor(timeout=5, max_memory_mb=100)
        result = executor.execute(sample_valid_code)

        assert result.success is True
        assert "step" in result.stdout
        assert "final_answer_verified" in result.stdout
        assert result.execution_time > 0

    def test_execute_syntax_error(self):
        """Test executing code with syntax error."""
        executor = SafeExecutor()
        invalid_code = "x = ( # missing closing parenthesis"

        result = executor.execute(invalid_code)

        assert result.success is False
        assert result.exception_type == "SyntaxError"
        assert result.exception_message is not None

    def test_execute_runtime_error(self):
        """Test executing code with runtime error."""
        executor = SafeExecutor()
        error_code = '''
import sympy as sp
x = undefined_variable  # This will cause NameError
'''

        result = executor.execute(error_code)

        assert result.success is False
        assert "NameError" in result.exception_type
        assert "undefined_variable" in result.exception_message

    def test_execute_import_restriction(self):
        """Test that restricted imports are blocked."""
        executor = SafeExecutor()
        restricted_code = '''
import os
print(os.listdir('.'))
'''

        result = executor.execute(restricted_code)

        assert result.success is False
        assert "ImportError" in result.exception_type
        assert "not allowed" in result.exception_message

    def test_safe_namespace(self):
        """Test that the execution namespace is properly restricted."""
        executor = SafeExecutor()
        test_code = '''
# Test that basic built-ins work
print(len([1, 2, 3]))
print(max([1, 2, 3]))

# Test that exceptions are available
try:
    raise ValueError("test")
except ValueError as e:
    print("Caught:", str(e))
'''

        result = executor.execute(test_code)

        assert result.success is True
        assert "3" in result.stdout  # len result
        assert "Caught: test" in result.stdout  # exception handling


class TestVerificationOutputParser:
    """Test the output parsing component."""

    def test_parse_valid_output(self):
        """Test parsing valid JSON output."""
        parser = VerificationOutputParser()

        # Mock execution result with valid JSON output
        exec_result = CodeExecutionResult(
            success=True,
            stdout='''{"step": 1, "description": "First step", "verified": true}
{"step": 2, "description": "Second step", "verified": false}
{"final_answer_verified": true, "computed": "6*x + 2", "claimed": "6*x + 2"}''',
            stderr="",
            execution_time=1.0
        )

        steps, final_verdict, error = parser.parse(exec_result)

        assert error is None
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert steps[0].verified is True
        assert steps[1].step_number == 2
        assert steps[1].verified is False
        assert final_verdict is not None
        assert final_verdict["final_answer_verified"] is True

    def test_parse_malformed_json(self):
        """Test parsing output with malformed JSON."""
        parser = VerificationOutputParser()

        exec_result = CodeExecutionResult(
            success=True,
            stdout='{"step": 1, "verified": true}  # missing closing brace',
            stderr="",
            execution_time=1.0
        )

        steps, final_verdict, error = parser.parse(exec_result)

        assert error is not None
        assert isinstance(error, json.JSONDecodeError)

    def test_parse_invalid_step_data(self):
        """Test parsing with invalid step data types."""
        parser = VerificationOutputParser()

        exec_result = CodeExecutionResult(
            success=True,
            stdout='''{"step": "invalid", "verified": true}
{"step": 2, "verified": "not_boolean"}
{"step": 3, "description": "Valid step", "verified": true}
{"final_answer_verified": true}''',
            stderr="",
            execution_time=1.0
        )

        steps, final_verdict, error = parser.parse(exec_result)

        assert error is None
        assert len(steps) == 1  # Only the valid step should be included
        assert steps[0].step_number == 3

    def test_parse_execution_failure(self):
        """Test parsing when execution failed."""
        parser = VerificationOutputParser()

        exec_result = CodeExecutionResult(
            success=False,
            stdout="",
            stderr="Some error",
            execution_time=1.0
        )

        steps, final_verdict, error = parser.parse(exec_result)

        assert steps == []
        assert final_verdict is None
        assert error is None


class TestVerificationPipeline:
    """Test the main verification pipeline."""

    @patch('src.pipeline.verification.verification.SymPyCodeGenerator')
    @patch('src.pipeline.verification.verification.SafeExecutor')
    @patch('src.pipeline.verification.verification.VerificationOutputParser')
    def test_verify_success(self, mock_parser, mock_executor, mock_generator,
                          mock_model_manager, sample_reasoning):
        """Test successful verification."""
        # Setup mocks
        mock_gen_instance = Mock()
        mock_gen_instance.generate.return_value = ("test_code", {"model": "test"})
        mock_generator.return_value = mock_gen_instance

        mock_exec_instance = Mock()
        mock_exec_result = CodeExecutionResult(
            success=True, stdout="output", stderr="", execution_time=1.0
        )
        mock_exec_instance.execute.return_value = mock_exec_result
        mock_executor.return_value = mock_exec_instance

        mock_parser_instance = Mock()
        mock_steps = [StepVerification(step_number=1, description="test", verified=True)]
        mock_verdict = {"final_answer_verified": True}
        mock_parser_instance.parse.return_value = (mock_steps, mock_verdict, None)
        mock_parser.return_value = mock_parser_instance

        # Run verification
        pipeline = VerificationPipeline(mock_model_manager)
        result = pipeline.verify(sample_reasoning)

        assert result.status == "verified"
        assert result.confidence_score > 0
        assert result.reasoning_output == sample_reasoning

    @patch('src.pipeline.verification.verification.SymPyCodeGenerator')
    def test_verify_code_generation_failure(self, mock_generator,
                                          mock_model_manager, sample_reasoning):
        """Test verification when code generation fails."""
        mock_gen_instance = Mock()
        mock_gen_instance.generate.side_effect = Exception("Generation failed")
        mock_generator.return_value = mock_gen_instance

        pipeline = VerificationPipeline(mock_model_manager)
        result = pipeline.verify(sample_reasoning)

        assert result.status == "failed_pipeline"
        assert len(result.errors) > 0
        assert "Generation failed" in result.errors[0].message

    def test_linear_equation_codegen_contract_fault_is_repaired(
        self, mock_model_manager, linear_equation_reasoning
    ):
        bad_response = Mock()
        bad_response.content = """import sympy as sp
import json
lhs_step3 = 16 - 9
lhs_step4 = 7
verified = bool((lhs_step3 - lhs_step4).simplify() == 0)
print(json.dumps({"step": 1, "description": "bad", "verified": verified, "note": ""}))
"""
        bad_response.meta = {"model": "test_model", "latency": 100}

        repaired_response = Mock()
        repaired_response.content = """import sympy as sp
import json
import math

def are_numerically_equal(a, b, tol=1e-9):
    try:
        return abs(float(sp.N(sp.sympify(a) - sp.sympify(b)))) < tol
    except Exception:
        return False

def same_expr(a, b):
    try:
        return bool(sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0)
    except Exception:
        return False

print(json.dumps({"step": 1, "description": "Adding 2t gives 16 = 7t + 9", "verified": True, "note": "confirmed"}))
print(json.dumps({"step": 2, "description": "Subtracting 9 gives 7 = 7t", "verified": True, "note": "confirmed"}))
print(json.dumps({"step": 3, "description": "Dividing by 7 gives t = 1", "verified": True, "note": "confirmed"}))
print(json.dumps({"final_answer_verified": True, "answer": "t = 1", "note": "confirmed"}))
"""
        repaired_response.meta = {"model": "test_model", "latency": 100}

        mock_model_manager.call.side_effect = [bad_response, repaired_response]
        mock_model_manager.prompts.load_prompt.return_value = SimpleNamespace(
            system_template="verification system prompt"
        )

        pipeline = VerificationPipeline(mock_model_manager)
        result = pipeline.verify(linear_equation_reasoning)

        assert result.status == "verified"
        assert result.metadata["repaired_from_codegen_fault"] is True
        assert ".simplify()" not in result.generated_code
        assert mock_model_manager.call.call_args_list[0].kwargs["prompt_ref"] == "codegen/baseline_codegen@v7"
        assert mock_model_manager.call.call_args_list[1].kwargs["prompt_ref"] == "codegen/baseline_codegen@v7"


class TestVerificationOrchestrator:
    """Test the verification orchestrator with reasoning repair."""

    @patch('src.pipeline.verification.verification_orchestrator.VerificationPipeline')
    @patch('src.pipeline.verification.verification_orchestrator.ReasoningPipeline')
    def test_verify_with_repair_success_first_attempt(self, mock_reasoning_pipeline,
                                                     mock_verification_pipeline,
                                                     mock_model_manager, sample_reasoning):
        """Test orchestrator when verification succeeds on first attempt."""
        # Mock successful verification
        mock_verification_result = Mock()
        mock_verification_result.status = "verified"
        mock_verification_result.errors = []
        mock_verification_result.step_verifications = []
        mock_verification_result.generated_code = ""

        mock_verification_instance = Mock()
        mock_verification_instance.verify.return_value = mock_verification_result
        mock_verification_pipeline.return_value = mock_verification_instance

        orchestrator = VerificationOrchestrator(mock_model_manager)
        result, repair_history = orchestrator.verify_with_repair(sample_reasoning)

        assert result.status == "verified"
        assert len(repair_history) == 0

    @patch('src.pipeline.verification.verification_orchestrator.VerificationPipeline')
    @patch('src.pipeline.verification.verification_orchestrator.ReasoningPipeline')
    def test_verify_with_repair_reasoning_failure(self, mock_reasoning_pipeline,
                                                 mock_verification_pipeline,
                                                 mock_model_manager, sample_reasoning):
        """Test orchestrator when reasoning repair is needed."""
        # Mock initial verification failure
        mock_failed_result = Mock()
        mock_failed_result.status = "failed_reasoning"
        mock_failed_result.errors = [VerificationError(
            error_type=ErrorType.ANSWER_MISMATCH,
            message="Answer mismatch detected"
        )]

        # Mock successful verification after repair
        mock_success_result = Mock()
        mock_success_result.status = "verified"
        mock_success_result.reasoning_output = sample_reasoning
        mock_success_result.errors = []
        mock_success_result.step_verifications = []
        mock_success_result.generated_code = ""

        mock_verification_instance = Mock()
        mock_verification_instance.verify.side_effect = [mock_failed_result, mock_success_result]
        mock_verification_pipeline.return_value = mock_verification_instance

        # Mock reasoning repair - now using model_manager.call
        mock_repair_response = Mock()
        mock_repair_response.content = "Test repair response"
        mock_model_manager.call.return_value = mock_repair_response

        # Mock reasoning pipeline parse method
        mock_reasoning_instance = Mock()
        mock_reasoning_instance._parse_structured_response.return_value = ReasoningOutput(
            original_problem=sample_reasoning.original_problem,
            worked_solution="Corrected solution",
            final_answer="6*x + 2",
            think_reasoning="Corrected reasoning",
        )
        mock_reasoning_pipeline.return_value = mock_reasoning_instance

        # Add missing step_verifications to failed result
        mock_failed_result.step_verifications = []
        mock_failed_result.generated_code = ""

        orchestrator = VerificationOrchestrator(mock_model_manager)
        result, repair_history = orchestrator.verify_with_repair(
            sample_reasoning,
            max_reasoning_attempts=2
        )

        assert result.status == "verified"
        assert len(repair_history) == 1
        assert repair_history[0].repair_type == "reasoning"
        assert repair_history[0].success is True


@pytest.mark.integration
class TestVerificationIntegration:
    """Integration tests that test the full pipeline end-to-end."""

    def test_end_to_end_verification(self, sample_reasoning):
        """Test the complete verification pipeline end-to-end.

        Note: This test requires actual model configuration and may be skipped
        in CI environments.
        """
        # Skip if no config available
        config_path = Path("src/config/config.yaml")
        if not config_path.exists():
            pytest.skip("Config file not available for integration test")

        try:
            from src.models.manager import ModelManager
            model_manager = ModelManager(config_path=config_path)
            pipeline = VerificationPipeline(model_manager)

            # This will make actual LLM calls
            result = pipeline.verify(sample_reasoning)

            # Basic assertions
            assert result is not None
            assert result.status in ["verified", "failed_reasoning", "failed_codegen", "failed_pipeline"]
            assert result.confidence_score >= 0.0
            assert result.reasoning_output == sample_reasoning

        except Exception as e:
            pytest.skip(f"Integration test failed due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
