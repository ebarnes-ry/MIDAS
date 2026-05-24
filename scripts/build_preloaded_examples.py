#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.profiles import resolve_config_path, validate_config_environment
from src.models.manager import ModelManager
from src.pipeline.vision.preloaded_examples import EXAMPLE_INPUTS, write_preloaded_example
from src.pipeline.vision.types import VisionInput
from src.pipeline.vision.vision import VisionPipeline


EXAMPLE_DIR = PROJECT_ROOT / "midas-frontend" / "public" / "examples"


def build_examples(example_ids: list[str]) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = resolve_config_path()
    validate_config_environment(config_path)

    manager = ModelManager(config_path=config_path)
    pipeline = VisionPipeline(manager)

    for example_id in example_ids:
        info = EXAMPLE_INPUTS[example_id]
        image_path = EXAMPLE_DIR / info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        start = time.time()
        document = pipeline.process_input(
            VisionInput(file_path=str(image_path), file_type="image/png")
        )
        elapsed = time.time() - start
        output_path = write_preloaded_example(
            example_id=example_id,
            image_path=image_path,
            document=document,
            processing_metadata={
                "filename": info["file_name"],
                "processing_time": elapsed,
                "generated_from_config": str(config_path),
            },
        )
        print(f"Wrote {output_path} ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cached Marker/grouping results for MIDAS demo examples."
    )
    parser.add_argument(
        "examples",
        nargs="*",
        choices=sorted(EXAMPLE_INPUTS),
        help="Example ids to generate. Defaults to all examples.",
    )
    args = parser.parse_args()
    build_examples(args.examples or list(EXAMPLE_INPUTS))


if __name__ == "__main__":
    main()
