from src.pipeline.vision.preloaded_examples import (
    EXAMPLE_INPUTS,
    available_preloaded_examples,
    load_preloaded_example,
    serialize_ui_document,
    deserialize_ui_document,
)


EXPECTED_PROBLEM_TEXT = {
    "definite-integral": "Evaluate the definite integral: $$\\int_0^2 \\left(3x^2 - 2x + 1\\right)\\,dx$$",
    "eigenvalues": "Find the eigenvalues of the matrix $$A = \\begin{bmatrix} 4 & 1 \\\\ 2 & 3 \\end{bmatrix}$$",
    "integration-by-parts": "Evaluate the integral: $$\\int x\\cos(x)\\,dx$$",
    "product-rule": "Find the derivative of \\(f(x) = x^2 e^x\\).",
    "quadratic-with-discriminant": "Solve \\(2x^2 - 7x + 3 = 0\\).",
    "system-linear-equations": "Solve the system: $$\\begin{cases}3x + 2y = 12 \\\\ x - y = 1\\end{cases}$$",
}


def test_preloaded_examples_exist_and_have_problem_text():
    available = {
        example["id"]: example["cached"]
        for example in available_preloaded_examples()
    }

    assert set(available) == set(EXAMPLE_INPUTS)
    assert all(available.values())

    for example_id in EXAMPLE_INPUTS:
        payload = load_preloaded_example(example_id)
        document = payload["ui_document"]

        assert payload["processing_metadata"]["cached"] is True
        assert payload["processing_metadata"]["source"] == "preloaded_example"
        assert document.blocks
        assert document.problems
        problem = document.problems[0]
        assert problem.problem_text == EXPECTED_PROBLEM_TEXT[example_id]
        assert problem.problem_input_complete is True
        assert problem.missing_problem_content is False
        assert problem.missing_content_reason is None
        assert "\\\\n" not in problem.problem_text
        assert "\n\\[" not in problem.problem_text
        assert "\n\\]" not in problem.problem_text
        assert problem.block_ids
        assert set(problem.block_ids).issubset({block.id for block in document.blocks})


def test_preloaded_document_serialization_round_trips():
    payload = load_preloaded_example("quadratic-with-discriminant")
    document = payload["ui_document"]

    round_tripped = deserialize_ui_document(serialize_ui_document(document))

    assert round_tripped.dimensions == document.dimensions
    assert len(round_tripped.blocks) == len(document.blocks)
    assert len(round_tripped.problems) == len(document.problems)
    assert round_tripped.problems[0].problem_text == document.problems[0].problem_text
    assert round_tripped.problems[0].problem_type == document.problems[0].problem_type
