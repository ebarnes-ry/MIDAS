from typing import Optional, Dict, Any, Tuple
import ast
import re

from src.models.manager import ModelManager
from ..reasoning.types import ReasoningOutput


class CodegenContractError(ValueError):
    def __init__(self, message: str, code: str, metadata: Dict[str, Any], category: str = "contract"):
        super().__init__(message)
        self.code = code
        self.metadata = metadata
        self.category = category


class CodegenContractValidationError(ValueError):
    def __init__(self, message: str, category: str):
        super().__init__(message)
        self.category = category


class SymPyCodeGenerator:
    """
    Generates SymPy verification code from reasoning outputs using a strict prompt.
    """

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def _task_prompt_ref(self, default: str = "codegen/baseline_codegen@v7") -> str:
        config = getattr(self.model_manager, "config", {}) or {}
        return (
            config.get("tasks", {})
            .get("verification", {})
            .get("prompt_ref")
            or default
        )

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

    def validate_code_contract(self, code: str) -> None:
        tree = ast.parse(code)

        parents: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def enclosing_function_name(node: ast.AST) -> Optional[str]:
            current = parents.get(node)
            while current is not None:
                if isinstance(current, ast.FunctionDef):
                    return current.name
                current = parents.get(current)
            return None

        function_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        missing_helpers = {"emit_step", "emit_final"} - function_names
        if missing_helpers:
            raise CodegenContractValidationError(
                "Generated verification code violates the v7 contract: "
                f"missing required helper(s): {', '.join(sorted(missing_helpers))}.",
                category="missing_helper",
            )

        emit_final_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "simplify":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "sp":
                        continue
                    raise CodegenContractValidationError(
                        "Generated verification code violates the v7 contract: "
                        "do not call .simplify() as an instance method; use "
                        "sp.simplify(sp.sympify(a) - sp.sympify(b)) or same_expr(a, b).",
                        category="simplify_instance_method",
                    )
                if (
                    node.func.attr == "dumps"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"
                    and enclosing_function_name(node) not in {"emit_step", "emit_final"}
                ):
                    raise CodegenContractValidationError(
                        "Generated verification code violates the v7 contract: "
                        "do not call json.dumps directly outside emit_step/emit_final; "
                        "all JSON output must pass through the emit helpers.",
                        category="forbidden_json_dumps",
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "emit_final":
                    emit_final_calls += 1

        if emit_final_calls != 1:
            raise CodegenContractValidationError(
                "Generated verification code violates the v7 contract: "
                "the code must contain exactly one emit_final(...) call.",
                category="emit_final_count",
            )

    def generate(self, reasoning: ReasoningOutput) -> Tuple[str, Dict[str, Any]]:
        """
        Generates the verification code by calling the configured LLM task.

        Returns:
            A tuple containing the generated code string and metadata about the call.
        """
        if not self.model_manager:
            raise ValueError("ModelManager is required to generate code.")

        try:
            prompt_ref = self._task_prompt_ref()
            response = self.model_manager.call(
                task="verification",
                prompt_ref=prompt_ref,
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
            "prompt_ref": prompt_ref,
        }
        try:
            self.validate_code_contract(code)
        except SyntaxError as e:
            raise CodegenContractError(str(e), code, metadata, category="syntax") from e
        except CodegenContractValidationError as e:
            raise CodegenContractError(str(e), code, metadata, category=e.category) from e

        return code, metadata
