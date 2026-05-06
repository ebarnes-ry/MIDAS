from typing import Optional, Dict, Any, Tuple
import re

from src.models.manager import ModelManager
from ..reasoning.types import ReasoningOutput


class SymPyCodeGenerator:
    """
    Generates SymPy verification code from reasoning outputs using a strict prompt.
    """

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def extract_code(self, model_response: str) -> Optional[str]:
        """
        Extracts Python code from a model's response, typically from a markdown block.
        """
        patterns = [
            r'```python\n(.*?)```',  # Standard python block
            r'```\n(.*?)```',        # Generic code block
        ]

        for pattern in patterns:
            match = re.search(pattern, model_response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback if no markdown block is found, but only if it looks like code
        # Be more restrictive to avoid returning arbitrary text as code
        if ('import sympy' in model_response and
            'import json' in model_response and
            len(model_response.strip().split('\n')) > 3):
            return model_response.strip()

        return None

    def generate(self, reasoning: ReasoningOutput) -> Tuple[str, Dict[str, Any]]:
        """
        Generates the verification code by calling the configured LLM task.

        Returns:
            A tuple containing the generated code string and metadata about the call.
        """
        if not self.model_manager:
            raise ValueError("ModelManager is required to generate code.")

        try:
            response = self.model_manager.call(
                task="verification",
                prompt_ref="codegen/baseline_codegen@v6",
                variables={"reasoning": reasoning},
            )
        except Exception as e:
            raise RuntimeError(f"LLM call for code generation failed: {e}")

        code = self.extract_code(response.content)
        if not code:
            raise ValueError("No valid Python code block found in the model's response.")

        metadata = {
            "model_used": response.meta.get("model"),
            "latency_ms": response.meta.get("latency"),
            "prompt_ref": "codegen/baseline_codegen@v6"
        }

        return code, metadata
