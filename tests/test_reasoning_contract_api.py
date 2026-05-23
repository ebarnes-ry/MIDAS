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
from src.pipeline.reasoning.types import ReasoningOutput, ReasoningStep
from src.pipeline.verification.verification_orchestrator import RepairAttempt
from src.pipeline.verification.verification_types import VerificationResult
from src.pipeline.vision.types import Problem, ProblemType, UIBlock, UIDocument, VisionFinalOutput, VisualContext
from src.pipeline.vision.vision import VisionPipeline
from src.api.routers.vision import limiter as vision_limiter


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


def test_complete_uses_repaired_reasoning_fields_when_repair_succeeds():
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

    initial_reasoning = ReasoningOutput(
        original_problem="Solve x + 1 = 2.",
        steps=[ReasoningStep(1, "x = 3", "bad algebra", "x = 3")],
        final_answer="3",
        think_reasoning="initial trace",
        processing_metadata={"attempt": "initial"},
    )
    repaired_reasoning = ReasoningOutput(
        original_problem="Solve x + 1 = 2.",
        steps=[ReasoningStep(1, "x = 1", "Subtract 1 from both sides.", "x = 1")],
        final_answer="1",
        think_reasoning="repaired trace",
        processing_metadata={"attempt": "repaired"},
    )
    verification_result = VerificationResult(
        status="verified",
        confidence_score=1.0,
        reasoning_output=repaired_reasoning,
        generated_code="",
        answer_match=True,
        errors=[],
        metadata={"final_verdict": {"final_answer_verified": True, "answer": "1", "note": ""}},
    )
    repair_history = [
        RepairAttempt(
            attempt_number=1,
            repair_type="reasoning",
            reason="Reasoning verification failed with status: failed_reasoning",
            success=True,
            processing_time=0.1,
            repaired_reasoning=repaired_reasoning,
        )
    ]

    with (
        patch("src.api.routers.vision.VisionPipeline.process_selection") as process_selection,
        patch("src.api.routers.vision.ReasoningPipeline.process") as process_reasoning,
        patch("src.api.routers.vision.VerificationOrchestrator") as orchestrator_class,
    ):
        process_selection.return_value = VisionFinalOutput(
            problem_statement="Solve x + 1 = 2.",
            visual_context=None,
            source_metadata={"problem_type": "algebra"},
        )
        process_reasoning.return_value = initial_reasoning
        orchestrator = Mock()
        orchestrator.verify_with_repair.return_value = (verification_result, repair_history)
        orchestrator_class.return_value = orchestrator

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
    assert payload["success"] is True
    assert payload["data"]["reasoning"]["final_answer"] == "1"
    assert payload["data"]["reasoning"]["worked_solution"].startswith("1. x = 1")
    assert payload["data"]["reasoning"]["think_reasoning"] == "repaired trace"
    assert payload["data"]["reasoning"]["metadata"]["attempt"] == "repaired"
    assert payload["data"]["verification"]["final_answer"] == "1"
    assert payload["data"]["verification"]["repair_history"][0]["final_answer"] == "3"
    assert payload["data"]["verification"]["repair_history"][1]["final_answer"] == "1"


def test_complete_passes_user_visual_context_override_to_reasoning():
    app = create_app()
    model_manager = Mock(spec=ModelManager)
    model_manager.config = {}
    app.dependency_overrides[get_model_manager] = lambda: model_manager

    png = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(png, format="PNG")
    image_base64 = base64.b64encode(png.getvalue()).decode("ascii")
    ui_document = UIDocument(
        blocks=[],
        full_page_text="Use the graph shown to find x.",
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

    reasoning_output = ReasoningOutput(
        original_problem="Use the graph shown to find x.",
        steps=[ReasoningStep(1, "x = 2", "Read from graph.", "x = 2")],
        final_answer="2",
        think_reasoning="",
        processing_metadata={},
    )
    verification_result = VerificationResult(
        status="verified",
        confidence_score=1.0,
        reasoning_output=reasoning_output,
        generated_code="",
        answer_match=True,
        errors=[],
        metadata={"final_verdict": {"final_answer_verified": True, "answer": "2", "note": ""}},
    )

    with (
        patch("src.api.routers.vision.VisionPipeline.process_selection") as process_selection,
        patch("src.api.routers.vision.ReasoningPipeline.process") as process_reasoning,
        patch("src.api.routers.vision.VerificationOrchestrator") as orchestrator_class,
    ):
        process_selection.return_value = VisionFinalOutput(
            problem_statement="Use the graph shown to find x.",
            visual_context=VisualContext(
                elements=[],
                summary="Edited graph context.",
                contains_essential_info=True,
            ),
            source_metadata={
                "problem_type": "algebra",
                "visual_context_required": True,
                "visual_context_attached": True,
                "visual_context_source": "user_override",
            },
        )
        process_reasoning.return_value = reasoning_output
        orchestrator = Mock()
        orchestrator.verify_with_repair.return_value = (verification_result, [])
        orchestrator_class.return_value = orchestrator

        response = client.post(
            "/api/v1/vision/complete",
            json={
                "document_id": "doc-1",
                "problem_id": "problem-1",
                "edited_latex": "Use the graph shown to find x.",
                "visual_context_override": "Edited graph context.",
                "remove_visual_context": False,
            },
        )

    assert response.status_code == 200
    process_selection.assert_called_once()
    assert process_selection.call_args.kwargs["visual_context_override"] == "Edited graph context."
    assert process_selection.call_args.kwargs["remove_visual_context"] is False
    reasoning_input = process_reasoning.call_args.args[0]
    assert reasoning_input.visual_context == "Edited graph context."
    payload = response.json()
    assert payload["data"]["vision"]["metadata"]["visual_context_source"] == "user_override"


def test_complete_filters_equation_fragment_visual_context_before_reasoning():
    app = create_app()
    app.state.limiter.enabled = False
    vision_limiter.enabled = False
    model_manager = Mock(spec=ModelManager)
    model_manager.config = {}
    app.dependency_overrides[get_model_manager] = lambda: model_manager

    problem = Problem(
        problem_id="problem-1",
        problem_text=r"$\int_0^2 (3x^2 - 2x + 1) \, dx$",
        block_ids=["eq-1"],
        problem_type=ProblemType.CALCULUS,
    )
    ui_document = UIDocument(
        blocks=[
            UIBlock(
                id="eq-1",
                block_type="Equation",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content=r"\int_0^2 (3x^2 - 2x + 1) \, dx",
                is_editable=True,
            ),
            UIBlock(
                id="picture-1",
                block_type="Picture",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                image_description=(
                    'The image shows bold black text "1) dx" on a light yellow background. '
                    'The text appears to be part of an equation notation with the differential dx.'
                ),
            ),
        ],
        full_page_text=problem.problem_text,
        images={},
        metadata={},
        dimensions=(10, 10),
        problems=[problem],
    )
    ui_document.problems = VisionPipeline.__new__(VisionPipeline)._associate_descriptions_to_problems(
        ui_document.problems,
        ui_document,
    )

    png = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(png, format="PNG")
    image_base64 = base64.b64encode(png.getvalue()).decode("ascii")
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

    reasoning_output = ReasoningOutput(
        original_problem=problem.problem_text,
        steps=[ReasoningStep(1, "Evaluate the integral.", "Use the power rule.", "")],
        final_answer="6",
        think_reasoning="",
        processing_metadata={},
    )
    verification_result = VerificationResult(
        status="verified",
        confidence_score=1.0,
        reasoning_output=reasoning_output,
        generated_code="",
        answer_match=True,
        errors=[],
        metadata={},
    )

    with (
        patch("src.api.routers.vision.ReasoningPipeline.process") as process_reasoning,
        patch("src.api.routers.vision.VerificationOrchestrator") as orchestrator_class,
    ):
        process_reasoning.return_value = reasoning_output
        orchestrator = Mock()
        orchestrator.verify_with_repair.return_value = (verification_result, [])
        orchestrator_class.return_value = orchestrator

        response = client.post(
            "/api/v1/vision/complete",
            json={
                "document_id": "doc-1",
                "problem_id": "problem-1",
                "edited_latex": problem.problem_text,
            },
        )

    assert response.status_code == 200
    reasoning_input = process_reasoning.call_args.args[0]
    assert reasoning_input.visual_context is None
    payload = response.json()
    assert payload["data"]["vision"]["visual_context"] is None
    assert payload["data"]["vision"]["metadata"]["visual_context_required"] is False
    assert payload["data"]["vision"]["metadata"]["visual_context_attached"] is False
    assert payload["data"]["vision"]["metadata"]["visual_context_description_count"] == 0
