"""
Tests for the verification API endpoints.

Tests the FastAPI verification endpoints including:
- Request/response validation
- Error handling
- Integration with verification orchestrator
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import create_app
from src.pipeline.verification.verification_types import VerificationResult, ErrorType, VerificationError
from src.pipeline.reasoning.types import ReasoningOutput
from src.models.manager import ModelManager


@pytest.fixture
def client():
    """Test client for the FastAPI app."""
    from src.api.dependencies.session import get_model_manager
    from src.api.main import create_app

    app = create_app()

    # Override the model manager dependency for testing
    def override_get_model_manager():
        return Mock(spec=ModelManager)

    app.dependency_overrides[get_model_manager] = override_get_model_manager
    return TestClient(app)


@pytest.fixture
def sample_verification_request():
    """Sample verification request payload."""
    return {
        "problem_statement": "Find the derivative of f(x) = 3x^2 + 2x + 1",
        "worked_solution": (
            "To find the derivative, we apply the power rule to each term.\n"
            "The derivative of 3x^2 is 6x.\n"
            "The derivative of 2x is 2.\n"
            "The derivative of 1 is 0.\n"
            "Therefore, f'(x) = 6x + 2."
        ),
        "final_answer": "6*x + 2",
        "think_reasoning": "This is a basic polynomial differentiation problem.",
        "enable_reasoning_repair": True,
        "max_reasoning_attempts": 2
    }


@pytest.fixture
def mock_verification_result():
    """Mock verification result for testing."""
    reasoning_output = ReasoningOutput(
        original_problem="Find the derivative of f(x) = 3x^2 + 2x + 1",
        worked_solution="Test solution",
        final_answer="6*x + 2",
        think_reasoning="Test reasoning",
        processing_metadata={}
    )

    return VerificationResult(
        status="verified",
        confidence_score=0.95,
        reasoning_output=reasoning_output,
        generated_code="import sympy\nx = sympy.Symbol('x')\nresult = 6*x + 2",
        answer_match=True,
        errors=[],
        metadata={}
    )


class TestVerificationAPI:
    """Test the verification API endpoints."""

    @patch('src.api.routers.verification.VerificationOrchestrator')
    def test_verify_success(self, mock_orchestrator_class, client,
                           sample_verification_request, mock_verification_result):
        """Test successful verification API call."""
        # Setup mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.verify_with_repair.return_value = (mock_verification_result, [])
        mock_orchestrator_class.return_value = mock_orchestrator

        # Make API request
        response = client.post("/api/v1/verification/verify",
                              json=sample_verification_request)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "verified"
        assert data["data"]["confidence_score"] == 0.95
        assert data["data"]["verification_passed"] is True
        assert data["data"]["reasoning_repair_attempts"] == 0

    @patch('src.api.routers.verification.VerificationOrchestrator')
    def test_verify_with_reasoning_repair(self, mock_orchestrator_class, client,
                                        sample_verification_request):
        """Test verification with reasoning repair."""
        from src.pipeline.verification.verification_orchestrator import RepairAttempt

        # Create mock results
        reasoning_output = ReasoningOutput(
            original_problem=sample_verification_request["problem_statement"],
            worked_solution="Corrected solution",
            final_answer="6*x + 2",
            think_reasoning="Corrected reasoning",
            processing_metadata={}
        )

        final_result = VerificationResult(
            status="verified",
            confidence_score=0.90,
            reasoning_output=reasoning_output,
            generated_code="corrected_code",
            answer_match=True,
            errors=[],
            metadata={}
        )

        repair_history = [
            RepairAttempt(
                attempt_number=1,
                repair_type="reasoning",
                reason="Answer mismatch detected",
                success=True,
                processing_time=2.5
            )
        ]

        # Setup mock
        mock_orchestrator = Mock()
        mock_orchestrator.verify_with_repair.return_value = (final_result, repair_history)
        mock_orchestrator_class.return_value = mock_orchestrator

        # Make API request
        response = client.post("/api/v1/verification/verify",
                              json=sample_verification_request)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["status"] == "verified"
        assert data["data"]["reasoning_repair_attempts"] == 1
        assert len(data["data"]["repair_history"]) == 1
        assert data["data"]["repair_history"][0]["type"] == "reasoning"
        assert data["data"]["repair_history"][0]["success"] is True

    @patch('src.api.routers.verification.VerificationOrchestrator')
    def test_verify_pipeline_failure(self, mock_orchestrator_class, client,
                                   sample_verification_request):
        """Test verification when the pipeline fails."""
        reasoning_output = ReasoningOutput(
            original_problem=sample_verification_request["problem_statement"],
            worked_solution=sample_verification_request["worked_solution"],
            final_answer=sample_verification_request["final_answer"],
            think_reasoning="",
            processing_metadata={}
        )

        failed_result = VerificationResult(
            status="failed_pipeline",
            confidence_score=0.0,
            reasoning_output=reasoning_output,
            generated_code="",
            answer_match=None,
            errors=[VerificationError(
                error_type=ErrorType.RUNTIME_ERROR,
                message="Code execution failed"
            )],
            metadata={"pipeline_failure": True}
        )

        mock_orchestrator = Mock()
        mock_orchestrator.verify_with_repair.return_value = (failed_result, [])
        mock_orchestrator_class.return_value = mock_orchestrator

        response = client.post("/api/v1/verification/verify",
                              json=sample_verification_request)

        assert response.status_code == 200  # API call succeeds
        data = response.json()

        assert data["success"] is True  # API success
        assert data["data"]["status"] == "failed_pipeline"
        assert data["data"]["confidence_score"] == 0.0
        assert data["data"]["verification_passed"] is False

    @patch('src.api.routers.verification.VerificationOrchestrator')
    def test_verify_codegen_failure_is_typed_non_500_response(
        self, mock_orchestrator_class, client, sample_verification_request
    ):
        """Expected verification failures should be returned as typed results."""
        reasoning_output = ReasoningOutput(
            original_problem=sample_verification_request["problem_statement"],
            worked_solution=sample_verification_request["worked_solution"],
            final_answer=sample_verification_request["final_answer"],
            think_reasoning="",
            processing_metadata={}
        )
        failed_result = VerificationResult(
            status="failed_codegen",
            confidence_score=0.0,
            reasoning_output=reasoning_output,
            generated_code="",
            answer_match=None,
            errors=[VerificationError(
                error_type=ErrorType.RUNTIME_ERROR,
                message="Initial code generation failed"
            )],
            metadata={"codegen_failure": True}
        )
        mock_orchestrator = Mock()
        mock_orchestrator.verify_with_repair.return_value = (failed_result, [])
        mock_orchestrator_class.return_value = mock_orchestrator

        response = client.post("/api/v1/verification/verify", json=sample_verification_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "failed_codegen"
        assert data["data"]["verification_passed"] is False
        assert data["data"]["processing_metadata"]["codegen_failure"] is True

    def test_verify_invalid_request(self, client):
        """Test verification with invalid request data."""
        invalid_request = {
            "problem_statement": "",  # Empty problem statement
            "worked_solution": "Some solution",
            # Missing required final_answer field
        }

        response = client.post("/api/v1/verification/verify",
                              json=invalid_request)

        assert response.status_code == 422  # Validation error

    @patch('src.api.routers.verification.VerificationOrchestrator')
    def test_verify_orchestrator_exception(self, mock_orchestrator_class, client,
                                         sample_verification_request):
        """Test verification when orchestrator raises exception."""
        mock_orchestrator = Mock()
        mock_orchestrator.verify_with_repair.side_effect = Exception("Orchestrator failed")
        mock_orchestrator_class.return_value = mock_orchestrator

        response = client.post("/api/v1/verification/verify",
                              json=sample_verification_request)

        assert response.status_code == 200  # API handles the exception
        data = response.json()

        assert data["success"] is False
        assert "Orchestrator failed" in data["message"]
        assert data["data"] is None

    def test_verify_request_validation(self, client):
        """Test request validation for verification endpoint."""
        # Test with all required fields
        valid_request = {
            "problem_statement": "Test problem",
            "worked_solution": "Test solution",
            "final_answer": "Test answer"
        }

        # Should not raise validation error
        with patch('src.api.routers.verification.VerificationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_verification_result = VerificationResult(
                status="verified",
                confidence_score=0.95,
                reasoning_output=Mock(),
                generated_code="test",
                answer_match=True,
                errors=[],
                metadata={}
            )
            mock_orchestrator.verify_with_repair.return_value = (mock_verification_result, [])
            mock_orchestrator_class.return_value = mock_orchestrator

            response = client.post("/api/v1/verification/verify", json=valid_request)
            assert response.status_code == 200

    def test_verify_optional_parameters(self, client):
        """Test verification with optional parameters."""
        request_with_options = {
            "problem_statement": "Test problem",
            "worked_solution": "Test solution",
            "final_answer": "Test answer",
            "think_reasoning": "Optional reasoning",
            "enable_reasoning_repair": False,
            "max_reasoning_attempts": 1,
            "source_metadata": {"test": "metadata"}
        }

        with patch('src.api.routers.verification.VerificationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_verification_result = VerificationResult(
                status="verified",
                confidence_score=0.95,
                reasoning_output=Mock(),
                generated_code="test",
                answer_match=True,
                errors=[],
                metadata={}
            )
            mock_orchestrator.verify_with_repair.return_value = (mock_verification_result, [])
            mock_orchestrator_class.return_value = mock_orchestrator

            response = client.post("/api/v1/verification/verify", json=request_with_options)
            assert response.status_code == 200

            # Verify the orchestrator was called with correct parameters
            call_args = mock_orchestrator.verify_with_repair.call_args
            # Since enable_reasoning_repair=False, max_attempts should be 0
            assert call_args[1]["max_reasoning_attempts"] == 0


class TestAPIEndpointIntegration:
    """Integration tests for the verification API."""

    def test_api_root_endpoint_includes_verification(self, client):
        """Test that the root endpoint includes verification in the endpoint list."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "verification" in data["endpoints"]
        assert data["endpoints"]["verification"] == "/api/v1/verification"

    def test_health_check_still_works(self, client):
        """Ensure adding verification doesn't break existing endpoints."""
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
