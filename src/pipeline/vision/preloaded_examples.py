from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.pipeline.vision.types import Problem, ProblemType, UIBlock, UIDocument


PRELOADED_EXAMPLE_DIR = Path(__file__).parents[2] / "data" / "preloaded_examples"

EXAMPLE_INPUTS = {
    "definite-integral": {
        "label": "Definite integral",
        "file_name": "definite-integral.png",
    },
    "product-rule": {
        "label": "Product rule",
        "file_name": "product-rule.png",
    },
    "integration-by-parts": {
        "label": "Integration by parts",
        "file_name": "integration-by-parts.png",
    },
    "eigenvalues": {
        "label": "Eigenvalues",
        "file_name": "eigenvalues.png",
    },
    "system-linear-equations": {
        "label": "Linear system",
        "file_name": "system-linear-equations.png",
    },
    "quadratic-with-discriminant": {
        "label": "Quadratic complex roots",
        "file_name": "quadratic-with-discriminant.png",
    },
}


def _serialize_block(block: UIBlock) -> Dict[str, Any]:
    data = asdict(block)
    data["children"] = [_serialize_block(child) for child in block.children]
    return data


def _deserialize_block(data: Dict[str, Any]) -> UIBlock:
    block_data = dict(data)
    block_data["children"] = [
        _deserialize_block(child) for child in block_data.get("children", [])
    ]
    return UIBlock(**block_data)


def _serialize_problem(problem: Problem) -> Dict[str, Any]:
    data = asdict(problem)
    data["problem_type"] = problem.problem_type.value
    return data


def _deserialize_problem(data: Dict[str, Any]) -> Problem:
    problem_data = dict(data)
    problem_data["problem_type"] = ProblemType(
        problem_data.get("problem_type") or ProblemType.OTHER.value
    )
    return Problem(**problem_data)


def serialize_ui_document(document: UIDocument) -> Dict[str, Any]:
    return {
        "blocks": [_serialize_block(block) for block in document.blocks],
        "full_page_text": document.full_page_text,
        "images": document.images,
        "metadata": document.metadata,
        "dimensions": list(document.dimensions),
        "problems": [_serialize_problem(problem) for problem in document.problems],
    }


def deserialize_ui_document(data: Dict[str, Any]) -> UIDocument:
    return UIDocument(
        blocks=[_deserialize_block(block) for block in data.get("blocks", [])],
        full_page_text=data.get("full_page_text", ""),
        images=data.get("images", {}),
        metadata=data.get("metadata", {}),
        dimensions=tuple(data.get("dimensions", (0, 0))),
        problems=[
            _deserialize_problem(problem) for problem in data.get("problems", [])
        ],
    )


def write_preloaded_example(
    *,
    example_id: str,
    image_path: Path,
    document: UIDocument,
    processing_metadata: Dict[str, Any],
    output_dir: Path = PRELOADED_EXAMPLE_DIR,
) -> Path:
    if example_id not in EXAMPLE_INPUTS:
        raise ValueError(f"Unknown example id: {example_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = image_path.read_bytes()
    payload = {
        "example_id": example_id,
        "label": EXAMPLE_INPUTS[example_id]["label"],
        "file_name": EXAMPLE_INPUTS[example_id]["file_name"],
        "original_image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "processing_metadata": {
            **processing_metadata,
            "cached": True,
            "source": "preloaded_example",
            "example_id": example_id,
            "example_label": EXAMPLE_INPUTS[example_id]["label"],
            "cache_note": (
                "This example uses a precomputed OCR and grouping result so the "
                "demo can skip the slow document-ingest step."
            ),
        },
        "ui_document": serialize_ui_document(document),
    }

    output_path = output_dir / f"{example_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_preloaded_example(
    example_id: str,
    input_dir: Path = PRELOADED_EXAMPLE_DIR,
) -> Dict[str, Any]:
    if example_id not in EXAMPLE_INPUTS:
        raise KeyError(example_id)

    path = input_dir / f"{example_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ui_document"] = deserialize_ui_document(payload["ui_document"])
    return payload


def available_preloaded_examples(input_dir: Path = PRELOADED_EXAMPLE_DIR) -> List[Dict[str, str]]:
    examples = []
    for example_id, info in EXAMPLE_INPUTS.items():
        examples.append(
            {
                "id": example_id,
                "label": info["label"],
                "file_name": info["file_name"],
                "cached": (input_dir / f"{example_id}.json").exists(),
            }
        )
    return examples
