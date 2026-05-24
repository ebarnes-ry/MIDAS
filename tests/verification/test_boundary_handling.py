from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from src.pipeline.reasoning.types import ReasoningOutput, ReasoningStep
from src.pipeline.verification.verification import VerificationPipeline
from src.pipeline.verification.verification_types import CodeExecutionResult
from src.pipeline.vision.types import Problem, ProblemType, UIDocument, UIBlock, UserSelection
from src.pipeline.vision.vision import VisionPipeline
from src.api.routers.vision import convert_ui_document_to_api_document
from src.pipeline.vision.grouper import SemanticGrouper


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


def test_abstract_proof_can_verify_when_symbolic_verification_succeeds():
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock(return_value=("test_code", {}))
    pipeline.executor.execute = Mock(
        return_value=CodeExecutionResult(
            success=True,
            stdout='{"step": 1, "description": "Check", "verified": true, "note": "ok"}\n'
                   '{"final_answer_verified": true, "answer": "", "note": "ok"}\n',
            stderr="",
            execution_time=0.01,
        )
    )
    reasoning = _reasoning(
        "Prove that there are infinitely many primes.",
        metadata={
            "problem_type": "proof",
            "visual_context_required": False,
            "visual_context_attached": False,
        },
    )

    result = pipeline.verify(reasoning)

    assert result.status == "verified"
    pipeline.code_generator.generate.assert_called_once()


def test_geometry_with_visual_context_attempts_verification_then_marks_codegen_failure_unsupported():
    pipeline = VerificationPipeline(_manager())
    pipeline.code_generator.generate = Mock(return_value=("x = undefined_variable", {}))
    pipeline.executor.execute = Mock(
        return_value=CodeExecutionResult(
            success=False,
            stdout="",
            stderr="NameError: name 'undefined_variable' is not defined",
            execution_time=0.01,
            exception_type="NameError",
            exception_message="name 'undefined_variable' is not defined",
        )
    )
    pipeline._get_repaired_code = Mock(side_effect=ValueError("Repair did not produce code."))
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
    assert result.metadata["unsupported_source"] == "post_verification_failure_boundary"
    assert result.metadata["underlying_verification_status"] == "failed_codegen"
    assert result.metadata["visual_context_attached"] is True
    pipeline.code_generator.generate.assert_called_once()


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


def test_instruction_stem_is_merged_with_immediately_following_equations():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    pipeline.grouper = SemanticGrouper.__new__(SemanticGrouper)
    problem = Problem(
        problem_id="problem_1",
        problem_text="Solve the system:",
        block_ids=["stem"],
        problem_type=ProblemType.ALGEBRA,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="stem",
                block_type="SectionHeader",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content="Solve the system:",
                is_editable=True,
            ),
            UIBlock(
                id="eq_1",
                block_type="Equation",
                html="",
                polygon=[0, 11, 10, 11, 10, 20, 0, 20],
                bbox=[0, 11, 10, 20],
                children=[],
                section_hierarchy={},
                latex_content="x + y = 3",
                is_editable=True,
            ),
            UIBlock(
                id="eq_2",
                block_type="Equation",
                html="",
                polygon=[0, 21, 10, 21, 10, 30, 0, 30],
                bbox=[0, 21, 10, 30],
                children=[],
                section_hierarchy={},
                latex_content="x - y = 1",
                is_editable=True,
            ),
        ],
        full_page_text="Solve the system:\n\nx + y = 3\n\nx - y = 1",
        images={},
        metadata={},
        dimensions=(10, 30),
        problems=[problem],
    )

    repaired = pipeline._repair_problem_assembly([problem], document)

    assert repaired[0].block_ids == ["stem", "eq_1", "eq_2"]
    assert repaired[0].problem_text == "Solve the system:\nx + y = 3\nx - y = 1"


def test_problem_type_classifier_detects_latex_math_domains():
    grouper = SemanticGrouper.__new__(SemanticGrouper)

    assert grouper._classify_problem_type(r"\int_0^1 x^2\,dx") == ProblemType.CALCULUS
    assert grouper._classify_problem_type(r"f'(x)=3x^2") == ProblemType.CALCULUS
    assert grouper._classify_problem_type(r"\begin{bmatrix}1&2\\3&4\end{bmatrix}") == ProblemType.LINEAR_ALGEBRA
    assert grouper._classify_problem_type(r"\begin{cases}x+y=3\\x-y=1\end{cases}") == ProblemType.ALGEBRA
    assert grouper._classify_problem_type(r"a \equiv b \pmod n") == ProblemType.NUMBER_THEORY


def test_problem_type_is_reclassified_using_linked_block_latex():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    pipeline.grouper = SemanticGrouper.__new__(SemanticGrouper)
    problem = Problem(
        problem_id="problem_1",
        problem_text="Evaluate",
        block_ids=["stem", "eq_1"],
        problem_type=ProblemType.OTHER,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="stem",
                block_type="Text",
                html="",
                polygon=[],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content="Evaluate",
                is_editable=True,
            ),
            UIBlock(
                id="eq_1",
                block_type="Equation",
                html="",
                polygon=[],
                bbox=[0, 11, 10, 20],
                children=[],
                section_hierarchy={},
                latex_content=r"\int_0^1 x^2\,dx",
                is_editable=True,
            ),
        ],
        full_page_text=r"Evaluate \int_0^1 x^2\,dx",
        images={},
        metadata={},
        dimensions=(10, 20),
        problems=[problem],
    )

    reclassified = pipeline._reclassify_problem_types([problem], document)

    assert reclassified[0].problem_type == ProblemType.CALCULUS


def test_instruction_stem_merge_stops_before_non_math_block():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    pipeline.grouper = SemanticGrouper.__new__(SemanticGrouper)
    problem = Problem(
        problem_id="problem_1",
        problem_text="Evaluate",
        block_ids=["stem"],
        problem_type=ProblemType.CALCULUS,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="stem",
                block_type="Text",
                html="",
                polygon=[],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content="Evaluate",
                is_editable=True,
            ),
            UIBlock(
                id="note",
                block_type="Text",
                html="",
                polygon=[],
                bbox=[0, 11, 10, 20],
                children=[],
                section_hierarchy={},
                latex_content="Use exact values.",
                is_editable=True,
            ),
            UIBlock(
                id="eq_1",
                block_type="Equation",
                html="",
                polygon=[],
                bbox=[0, 21, 10, 30],
                children=[],
                section_hierarchy={},
                latex_content=r"\int_0^1 x\,dx",
                is_editable=True,
            ),
        ],
        full_page_text=r"Evaluate Use exact values. \int_0^1 x\,dx",
        images={},
        metadata={},
        dimensions=(10, 30),
        problems=[problem],
    )

    repaired = pipeline._repair_problem_assembly([problem], document)

    assert repaired[0].block_ids == ["stem"]
    assert repaired[0].problem_text == "Evaluate"


def test_equation_fragment_descriptions_are_not_attached_as_visual_context():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    problem = Problem(
        problem_id="problem_1",
        problem_text=r"$\int_0^2 (3x^2 - 2x + 1) \, dx$",
        figure_references=["Picture 1"],
        block_ids=["eq_1"],
        problem_type=ProblemType.CALCULUS,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="eq_1",
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
                id="picture_1",
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

    document.problems = pipeline._associate_descriptions_to_problems(document.problems, document)
    output = pipeline.process_selection(
        UserSelection(
            problem_id="problem_1",
            edited_latex=problem.problem_text,
            original_image_path="",
        ),
        document,
        Image.new("RGB", (10, 10), "white"),
    )

    assert document.problems[0].referenced_figure_descriptions == []
    assert document.problems[0].figure_references == []
    assert output.visual_context is None
    assert output.source_metadata["visual_context_required"] is False
    assert output.source_metadata["visual_context_attached"] is False


def test_referenced_graph_description_is_still_attached_as_visual_context():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    problem = Problem(
        problem_id="problem_1",
        problem_text="Use Figure 1 to find the x-intercept of the graph.",
        figure_references=["Figure 1"],
        block_ids=["text_1"],
        problem_type=ProblemType.ALGEBRA,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="text_1",
                block_type="Text",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content="Use Figure 1 to find the x-intercept of the graph.",
                is_editable=True,
            ),
            UIBlock(
                id="figure_1",
                block_type="Figure",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                image_description="A coordinate plane graph with a line crossing the x-axis at x = 3.",
            ),
        ],
        full_page_text=problem.problem_text,
        images={},
        metadata={},
        dimensions=(10, 10),
        problems=[problem],
    )

    document.problems = pipeline._associate_descriptions_to_problems(document.problems, document)

    assert document.problems[0].referenced_figure_descriptions == [
        "A coordinate plane graph with a line crossing the x-axis at x = 3."
    ]


def test_api_problem_metadata_uses_filtered_visual_context_not_raw_marker_noise():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    problem = Problem(
        problem_id="problem_1",
        problem_text=r"$\int_0^2 x^2 e^x \, dx$",
        block_ids=[],
        problem_type=ProblemType.CALCULUS,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="eq_1",
                block_type="Equation",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content=r"x^2 e^x",
                is_editable=True,
            ),
            UIBlock(
                id="picture_1",
                block_type="Picture",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                image_description="The image shows duplicated mathematical expression x^2 e^x.",
            ),
        ],
        full_page_text=problem.problem_text,
        images={},
        metadata={},
        dimensions=(10, 10),
        problems=[problem],
    )

    document.problems = pipeline._associate_descriptions_to_problems(document.problems, document)
    api_document = convert_ui_document_to_api_document(document, Image.new("RGB", (10, 10), "white"))

    assert api_document.blocks[1].image_description == "The image shows duplicated mathematical expression x^2 e^x."
    assert api_document.problems[0].visual_context_attached is False
    assert api_document.problems[0].visual_context_summary is None
    assert api_document.problems[0].visual_context_description_count == 0


def test_plain_numeral_description_with_graphics_word_is_not_visual_context():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    problem = Problem(
        problem_id="problem_1",
        problem_text=r"$2x^2 - 7x + 3 = 0$",
        figure_references=["Picture 1"],
        block_ids=["eq_1"],
        problem_type=ProblemType.ALGEBRA,
    )
    document = UIDocument(
        blocks=[
            UIBlock(
                id="eq_1",
                block_type="Equation",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                latex_content=r"2x^2 - 7x + 3 = 0",
                is_editable=True,
            ),
            UIBlock(
                id="picture_1",
                block_type="Picture",
                html="",
                polygon=[0, 0, 10, 0, 10, 10, 0, 10],
                bbox=[0, 0, 10, 10],
                children=[],
                section_hierarchy={},
                image_description=(
                    'The image consists of a large, bold numeral "0" centered on a white '
                    "background without any other text or graphics."
                ),
            ),
        ],
        full_page_text=problem.problem_text,
        images={},
        metadata={},
        dimensions=(10, 10),
        problems=[problem],
    )

    document.problems = pipeline._associate_descriptions_to_problems(document.problems, document)
    output = pipeline.process_selection(
        UserSelection(
            problem_id="problem_1",
            edited_latex=problem.problem_text,
            original_image_path="",
        ),
        document,
        Image.new("RGB", (10, 10), "white"),
    )

    assert document.problems[0].referenced_figure_descriptions == []
    assert document.problems[0].figure_references == []
    assert output.visual_context is None
    assert output.source_metadata["visual_context_required"] is False
    assert output.source_metadata["visual_context_attached"] is False


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
