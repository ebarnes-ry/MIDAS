from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from src.pipeline.reasoning.types import ReasoningOutput, ReasoningStep
from src.pipeline.verification.verification import VerificationPipeline
from src.pipeline.vision.types import Problem, ProblemType, UIDocument, UIBlock, UserSelection
from src.pipeline.vision.vision import VisionPipeline


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


def _reasoning(problem: str, *, metadata=None) -> ReasoningOutput:
    return ReasoningOutput(
        original_problem=problem,
        steps=[
            ReasoningStep(
                step_number=1,
                claim="A candidate solution is provided.",
                justification="Model-generated reasoning.",
                latex_expression="",
            )
        ],
        final_answer="",
        think_reasoning="",
        processing_metadata=metadata or {},
    )


def test_missing_visual_context_returns_needs_visual_context_before_codegen():
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock()
    reasoning = _reasoning(
        "Use the diagram below to find x.",
        metadata={
            "problem_type": "geometry",
            "visual_context_required": True,
            "visual_context_attached": False,
        },
    )

    result = pipeline.verify(reasoning)

    assert result.status == "needs_visual_context"
    assert result.metadata["needs_visual_context"] is True
    assert result.metadata["unsupported_reason"] == "missing_visual_context"
    pipeline.code_generator.generate.assert_not_called()


def test_abstract_proof_returns_unsupported_before_codegen():
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock()
    reasoning = _reasoning(
        "Prove that there are infinitely many primes.",
        metadata={
            "problem_type": "proof",
            "visual_context_required": False,
            "visual_context_attached": False,
        },
    )

    result = pipeline.verify(reasoning)

    assert result.status == "unsupported"
    assert result.metadata["unsupported_reason"] == "abstract_proof_verification_boundary"
    pipeline.code_generator.generate.assert_not_called()


def test_geometry_with_visual_context_is_marked_unsupported_not_codegen_failure():
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock()
    reasoning = _reasoning(
        "In the triangle shown, find the missing angle.",
        metadata={
            "problem_type": "geometry",
            "visual_context_required": True,
            "visual_context_attached": True,
        },
    )

    result = pipeline.verify(reasoning)

    assert result.status == "unsupported"
    assert result.metadata["unsupported_reason"] == "geometry_symbolic_verification_boundary"
    assert result.metadata["visual_context_attached"] is True
    pipeline.code_generator.generate.assert_not_called()


def test_simple_algebra_still_reaches_codegen():
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock(side_effect=RuntimeError("stop after gate"))
    reasoning = _reasoning(
        "Solve x + 1 = 2.",
        metadata={
            "problem_type": "algebra",
            "visual_context_required": False,
            "visual_context_attached": False,
        },
    )

    result = pipeline.verify(reasoning)

    assert result.status == "failed_codegen"
    pipeline.code_generator.generate.assert_called_once()


def test_visual_selection_metadata_records_attached_description_without_explicit_reference():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    problem = Problem(
        problem_id="problem_1",
        problem_text="In the triangle shown, find x.",
        problem_type=ProblemType.GEOMETRY,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="figure_1",
                block_type="Figure",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                image_description="A triangle with angles 50 degrees, 60 degrees, and x.",
            )
        ],
        full_page_text="In the triangle shown, find x.",
        images={},
        metadata={},
        dimensions=(10, 10),
        problems=[problem],
    )
    document.problems = pipeline._associate_descriptions_to_problems(document.problems, document)

    output = pipeline.process_selection(
        UserSelection(
            problem_id="problem_1",
            edited_latex="In the triangle shown, find x.",
            original_image_path="",
        ),
        document,
        Image.new("RGB", (10, 10), "white"),
    )

    assert output.visual_context is not None
    assert "triangle with angles" in output.visual_context.summary
    assert output.source_metadata["problem_type"] == "geometry"
    assert output.source_metadata["visual_context_required"] is True
    assert output.source_metadata["visual_context_attached"] is True
    assert output.source_metadata["visual_context_description_count"] == 1


def test_visual_selection_accepts_user_override_and_removal():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    problem = Problem(
        problem_id="problem_1",
        problem_text="Use the graph shown to find the intercept.",
        problem_type=ProblemType.ALGEBRA,
        figure_references=["graph"],
        referenced_figure_descriptions=["Original graph description."],
    )
    document = UIDocument(
        blocks=[],
        full_page_text=problem.problem_text,
        images={},
        metadata={},
        dimensions=(10, 10),
        problems=[problem],
    )
    selection = UserSelection(
        problem_id="problem_1",
        edited_latex=problem.problem_text,
        original_image_path="",
    )

    overridden = pipeline.process_selection(
        selection,
        document,
        Image.new("RGB", (10, 10), "white"),
        visual_context_override="Edited graph description.",
    )
    removed = pipeline.process_selection(
        selection,
        document,
        Image.new("RGB", (10, 10), "white"),
        remove_visual_context=True,
    )

    assert overridden.visual_context is not None
    assert overridden.visual_context.summary == "Edited graph description."
    assert overridden.source_metadata["visual_context_source"] == "user_override"
    assert overridden.source_metadata["visual_context_user_modified"] is True
    assert removed.visual_context is None
    assert removed.source_metadata["visual_context_attached"] is False
    assert removed.source_metadata["visual_context_source"] == "user_removed"
    assert removed.source_metadata["visual_context_missing_reason"] == "user_removed"
