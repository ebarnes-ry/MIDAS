from types import SimpleNamespace

import pytest

from src.models.services.marker import MarkerService
from src.pipeline.reasoning.reasoning import ReasoningContractError, ReasoningPipeline, ReasoningResponseSchema
from src.pipeline.reasoning.types import ReasoningInput, ReasoningOutput, ReasoningStep
from src.pipeline.verification.verification_orchestrator import VerificationOrchestrator
from src.pipeline.vision.grouper import SemanticGrouper


class RecordingManager:
    def __init__(self, config, response):
        self.config = config
        self.response = response
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs.get("schema")
        if schema:
            if schema is ReasoningResponseSchema:
                return SimpleNamespace(
                    parsed=schema(
                        given="Solve \\(x + 1 = 2\\).",
                        steps=[
                            {
                                "step_number": 1,
                                "claim": "\\(x = 1\\).",
                                "latex": "x = 1",
                                "justification": "Subtracting \\(1\\) from both sides.",
                            }
                        ],
                        answer={"value": "1", "latex": "1"},
                    ),
                    content='{"given":"Solve x + 1 = 2."}',
                    meta={"model": "test-model"},
                )
            return SimpleNamespace(
                parsed=schema(problems=[{"problem_text": "Solve x + 1 = 2", "figure_references": []}]),
                content="{}",
                meta={"model": "test-model"},
            )
        return self.response


def test_reasoning_pipeline_uses_configured_prompt_ref():
    response = SimpleNamespace(
        content=(
            "<thinking>scratch</thinking>"
            "<solution>"
            '<step number="1"><claim>x = 1</claim><latex>$x=1$</latex>'
            "<justification>subtract 1</justification></step>"
            "<answer><value>1</value><latex>$1$</latex></answer>"
            "</solution>"
        ),
        meta={"model": "test-model"},
    )
    manager = RecordingManager(
        {"tasks": {"reasoning": {"prompt_ref": "reasoning/solve@custom"}}},
        response,
    )

    output = ReasoningPipeline(manager).process(ReasoningInput(problem_statement="x + 1 = 2"))

    assert manager.calls[0]["prompt_ref"] == "reasoning/solve@custom"
    assert output.think_reasoning == "scratch"
    assert output.processing_metadata["prompt_version"] == "reasoning/solve@custom"


def test_hosted_openai_reasoning_uses_json_schema_contract():
    manager = RecordingManager(
        {"tasks": {"reasoning": {"provider": "openai", "prompt_ref": "reasoning/solve@v3"}}},
        SimpleNamespace(content="", meta={"model": "test-model"}),
    )

    output = ReasoningPipeline(manager).process(ReasoningInput(problem_statement="x + 1 = 2"))

    assert manager.calls[0]["prompt_ref"] == "reasoning/solve@v3"
    assert manager.calls[0]["schema"] is ReasoningResponseSchema
    assert output.think_reasoning == ""
    assert output.steps[0].claim == "\\(x = 1\\)."
    assert output.final_answer == "1"
    assert output.processing_metadata["reasoning_contract"] == "json_schema"


def test_local_reasoning_v2_keeps_xml_contract():
    response = SimpleNamespace(
        content=(
            "<solution>"
            '<step number="1"><claim>x = 1</claim><latex>x=1</latex>'
            "<justification>subtract 1</justification></step>"
            "<answer><value>1</value><latex>1</latex></answer>"
            "</solution>"
        ),
        meta={"model": "test-model"},
    )
    manager = RecordingManager(
        {"tasks": {"reasoning": {"provider": "ollama_local", "prompt_ref": "reasoning/solve@v2"}}},
        response,
    )

    output = ReasoningPipeline(manager).process(ReasoningInput(problem_statement="x + 1 = 2"))

    assert manager.calls[0]["prompt_ref"] == "reasoning/solve@v2"
    assert manager.calls[0]["schema"] is None
    assert "reasoning_contract" not in output.processing_metadata


def test_reasoning_pipeline_rejects_unstructured_prose(capsys):
    response = SimpleNamespace(
        content="1. Subtract 1 from both sides.\n2. Therefore x = 1.",
        meta={"model": "test-model"},
    )
    manager = RecordingManager({"tasks": {}}, response)

    with pytest.raises(ReasoningContractError, match="missing required <solution>"):
        ReasoningPipeline(manager).process(ReasoningInput(problem_statement="x + 1 = 2"))

    captured = capsys.readouterr()
    assert "=== REASONING CONTRACT FAILURE ===" in captured.out
    assert "Reasoning response missing required <solution> block." in captured.out
    assert "=== RAW REASONING MODEL OUTPUT BEGIN ===" in captured.out
    assert "1. Subtract 1 from both sides.\n2. Therefore x = 1." in captured.out
    assert "=== RAW REASONING MODEL OUTPUT END ===" in captured.out


def test_reasoning_pipeline_requires_answer_block():
    response = SimpleNamespace(
        content=(
            "<solution>"
            '<step number="1"><claim>x = 1</claim><latex>$x=1$</latex>'
            "<justification>subtract 1</justification></step>"
            "</solution>"
        ),
        meta={"model": "test-model"},
    )
    manager = RecordingManager({"tasks": {}}, response)

    with pytest.raises(ReasoningContractError, match="missing required <answer>"):
        ReasoningPipeline(manager).process(ReasoningInput(problem_statement="x + 1 = 2"))


def test_reasoning_pipeline_can_use_balanced_boxed_answer_helper():
    response = SimpleNamespace(
        content=(
            "<solution>"
            '<step number="1"><claim>x = \\frac{3}{5}</claim><latex>$x=\\frac{3}{5}$</latex>'
            "<justification>solve</justification></step>"
            "<answer><value></value><latex>$\\boxed{\\frac{3}{5}}$</latex></answer>"
            "</solution>"
        ),
        meta={"model": "test-model"},
    )
    manager = RecordingManager({"tasks": {}}, response)

    output = ReasoningPipeline(manager).process(ReasoningInput(problem_statement="solve"))

    assert output.final_answer == "\\frac{3}{5}"


def test_reasoning_output_accepts_legacy_worked_solution_constructor():
    output = ReasoningOutput(
        original_problem="x + 1 = 2",
        worked_solution="Subtract 1 from both sides, so x = 1.",
        final_answer="1",
        think_reasoning="",
    )

    assert len(output.steps) == 1
    assert output.steps[0].claim == "Subtract 1 from both sides, so x = 1."
    assert output.worked_solution == "1. Subtract 1 from both sides, so x = 1.\n   (legacy worked_solution input)"


def test_reasoning_output_prefers_typed_steps_over_legacy_worked_solution():
    output = ReasoningOutput(
        original_problem="x + 1 = 2",
        steps=[ReasoningStep(step_number=1, claim="x = 1", justification="subtract 1")],
        worked_solution="legacy text",
        final_answer="1",
        think_reasoning="",
    )

    assert len(output.steps) == 1
    assert output.steps[0].claim == "x = 1"
    assert "legacy text" not in output.worked_solution


def test_semantic_grouper_uses_configured_prompt_ref():
    manager = RecordingManager(
        {"tasks": {"group_problems": {"prompt_ref": "vision/group_problems@custom"}}},
        SimpleNamespace(content="", meta={}),
    )

    problems = SemanticGrouper(manager).group("Solve x + 1 = 2")

    assert manager.calls[0]["prompt_ref"] == "vision/group_problems@custom"
    assert problems[0].problem_text == "Solve x + 1 = 2"


def test_reasoning_repair_uses_existing_structured_parser():
    repaired = ReasoningOutput(
        original_problem="x + 1 = 2",
        steps=[ReasoningStep(step_number=1, claim="x = 1", justification="subtract 1")],
        final_answer="1",
        think_reasoning="",
    )

    class Parser:
        def __init__(self):
            self.called = False

        def _parse_structured_response(self, content, original_problem, response):
            self.called = True
            assert content == "repaired content"
            assert original_problem == "x + 1 = 2"
            return repaired

    manager = RecordingManager(
        {},
        SimpleNamespace(content="repaired content", meta={"model": "test-model"}),
    )
    parser = Parser()
    orchestrator = VerificationOrchestrator.__new__(VerificationOrchestrator)
    orchestrator.model_manager = manager
    orchestrator.reasoning_pipeline = parser

    failed_reasoning = ReasoningOutput(
        original_problem="x + 1 = 2",
        steps=[ReasoningStep(step_number=1, claim="x = 3", justification="bad algebra")],
        final_answer="3",
        think_reasoning="",
    )
    verification_result = SimpleNamespace(
        status="failed_reasoning",
        errors=[SimpleNamespace(message="Answer mismatch")],
        step_verifications=[],
    )

    result = orchestrator._attempt_reasoning_repair(failed_reasoning, verification_result)

    assert parser.called is True
    assert result is repaired


def test_hosted_reasoning_repair_uses_configured_json_schema_contract():
    manager = RecordingManager(
        {"tasks": {"reasoning_repair": {"provider": "openai", "prompt_ref": "reasoning/repair@v2"}}},
        SimpleNamespace(content="", meta={"model": "test-model"}),
    )
    orchestrator = VerificationOrchestrator.__new__(VerificationOrchestrator)
    orchestrator.model_manager = manager
    orchestrator.reasoning_pipeline = ReasoningPipeline(manager)

    failed_reasoning = ReasoningOutput(
        original_problem="x + 1 = 2",
        steps=[ReasoningStep(step_number=1, claim="\\(x = 3\\).", justification="bad algebra")],
        final_answer="3",
        think_reasoning="",
    )
    verification_result = SimpleNamespace(
        status="failed_reasoning",
        errors=[SimpleNamespace(message="Answer mismatch")],
        step_verifications=[],
    )

    result = orchestrator._attempt_reasoning_repair(failed_reasoning, verification_result)

    assert manager.calls[0]["task"] == "reasoning_repair"
    assert manager.calls[0]["prompt_ref"] == "reasoning/repair@v2"
    assert manager.calls[0]["schema"] is ReasoningResponseSchema
    assert result.final_answer == "1"
    assert result.processing_metadata["source"] == "reasoning_repair"
    assert result.processing_metadata["reasoning_contract"] == "json_schema"


def test_marker_does_not_require_gemini_when_llm_disabled(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    service = MarkerService.__new__(MarkerService)
    service.settings = {"use_llm": False}

    config = service._build_cli_config()

    assert config["use_llm"] is False
    assert "gemini_api_key" not in config


def test_marker_uses_gemini_api_key_when_llm_enabled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    service = MarkerService.__new__(MarkerService)
    service.settings = {
        "use_llm": True,
        "llm_service": "gemini",
        "gemini": {"model": "gemini-2.5-flash"},
    }

    config = service._build_cli_config()

    assert config["use_llm"] is True
    assert config["gemini_api_key"] == "gemini-key"
    assert config["gemini_model_name"] == "gemini-2.5-flash"


def test_marker_uses_openai_service_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    service = MarkerService.__new__(MarkerService)
    service.settings = {
        "use_llm": True,
        "llm_service": "openai",
        "openai": {
            "model": "gpt-4.1-mini",
            "base_url": "https://api.openai.com/v1",
            "image_format": "png",
            "timeout": 120,
        },
    }

    config = service._build_cli_config()

    assert config["use_llm"] is True
    assert config["llm_service"] == "marker.services.openai.OpenAIService"
    assert config["openai_api_key"] == "openai-key"
    assert config["openai_model"] == "gpt-4.1-mini"
    assert config["openai_base_url"] == "https://api.openai.com/v1"
    assert config["openai_image_format"] == "png"
    assert config["timeout"] == 120
