import base64
import io
from datetime import datetime
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from src.api.dependencies.session import DocumentSession, get_model_manager, get_session_manager
from src.api.main import create_app
from src.models.manager import ModelManager
from src.pipeline.reasoning.reasoning import ReasoningContractError
from src.pipeline.vision.types import UIDocument, VisionFinalOutput


def test_reasoning_contract_error_is_typed_non_500_response():
    app = create_app()
    app.dependency_overrides[get_model_manager] = lambda: Mock(spec=ModelManager)
    client = TestClient(app)
    raw_output = "raw model output should not be returned"

    with patch("src.api.routers.reasoning.ReasoningPipeline.process") as process:
        process.side_effect = ReasoningContractError(
            "Reasoning JSON schema response failed validation.",
            prompt_ref="reasoning/solve@v3",
            model="test-model",
            original_problem="Solve x + 1 = 2.",
            raw_output=raw_output,
        )

        response = client.post(
            "/api/v1/reasoning/reason",
            json={"problem_statement": "Solve x + 1 = 2."},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["message"] == "Reasoning output violated the structured contract"
    assert payload["data"]["processing_metadata"]["status"] == "failed_contract"
    assert payload["data"]["processing_metadata"]["error_type"] == "contract_violation"
    assert payload["data"]["processing_metadata"]["prompt_ref"] == "reasoning/solve@v3"
    assert payload["data"]["processing_metadata"]["raw_response_length"] == len(raw_output)
    assert raw_output not in response.text


def test_complete_contract_error_is_typed_non_500_response():
    app = create_app()
    model_manager = Mock(spec=ModelManager)
    model_manager.config = {}
    app.dependency_overrides[get_model_manager] = lambda: model_manager

    png = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(png, format="PNG")
    image_base64 = base64.b64encode(png.getvalue()).decode("ascii")
    ui_document = UIDocument(
        blocks=[],
        full_page_text="Solve x + 1 = 2.",
        images={},
        metadata={},
        dimensions=(2, 2),
        problems=[],
    )
    session = DocumentSession(
        document_id="doc-1",
        ui_document=ui_document,
        original_image_base64=image_base64,
        created_at=datetime.utcnow(),
        last_accessed=datetime.utcnow(),
        processing_metadata={},
    )
    session_manager = Mock()
    session_manager.get_session.return_value = session
    app.dependency_overrides[get_session_manager] = lambda: session_manager
    client = TestClient(app)
    raw_output = "raw complete output should not be returned"

    with (
        patch("src.api.routers.vision.VisionPipeline.process_selection") as process_selection,
        patch("src.api.routers.vision.ReasoningPipeline.process") as process_reasoning,
        patch("src.api.routers.vision.TrajectoryLogger") as trajectory_logger,
    ):
        process_selection.return_value = VisionFinalOutput(
            problem_statement="Solve x + 1 = 2.",
            visual_context=None,
            source_metadata={"problem_type": "algebra"},
        )
        process_reasoning.side_effect = ReasoningContractError(
            "Reasoning JSON schema response failed validation.",
            prompt_ref="reasoning/solve@v3",
            model="test-model",
            original_problem="Solve x + 1 = 2.",
            raw_output=raw_output,
        )

        response = client.post(
            "/api/v1/vision/complete",
            json={
                "document_id": "doc-1",
                "problem_id": "problem-1",
                "edited_latex": "Solve x + 1 = 2.",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["message"] == "Reasoning output violated the structured contract"
    assert payload["data"]["reasoning"]["metadata"]["status"] == "failed_contract"
    assert payload["data"]["verification"]["status"] == "failed_contract"
    assert payload["data"]["verification"]["metadata"]["contract_source"] == "reasoning_solve"
    assert payload["data"]["verification"]["metadata"]["raw_response_length"] == len(raw_output)
    assert raw_output not in response.text
    trajectory_logger.return_value.log_attempt.assert_called_once()
    trajectory_logger.return_value.close_trajectory.assert_called_once()
