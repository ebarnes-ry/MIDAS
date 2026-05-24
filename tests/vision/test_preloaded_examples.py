from src.pipeline.vision.preloaded_examples import (
    EXAMPLE_INPUTS,
    available_preloaded_examples,
    load_preloaded_example,
    serialize_ui_document,
    deserialize_ui_document,
)


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
        assert document.problems[0].problem_text.strip()


def test_preloaded_document_serialization_round_trips():
    payload = load_preloaded_example("quadratic-with-discriminant")
    document = payload["ui_document"]

    round_tripped = deserialize_ui_document(serialize_ui_document(document))

    assert round_tripped.dimensions == document.dimensions
    assert len(round_tripped.blocks) == len(document.blocks)
    assert len(round_tripped.problems) == len(document.problems)
    assert round_tripped.problems[0].problem_text == document.problems[0].problem_text
    assert round_tripped.problems[0].problem_type == document.problems[0].problem_type
