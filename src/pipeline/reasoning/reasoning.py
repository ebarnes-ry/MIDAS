import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.models.manager import ModelManager
from .types import ReasoningInput, ReasoningOutput, ReasoningStep

logger = logging.getLogger(__name__)

# Matches <think>, <Think>, <thinking>, <Thinking>, <Thought>, <thought>
# DeepSeek-R1 variants use different tag names depending on the distillation.
_THINK_RE = re.compile(
    r'<([Tt]hink(?:ing)?|[Tt]hought)>(.*?)</\1>',
    re.DOTALL,
)
_STRIP_THINK_RE = re.compile(
    r'<[Tt]hink(?:ing)?>.*?</[Tt]hink(?:ing)?>|<[Tt]hought>.*?</[Tt]hought>',
    re.DOTALL,
)


class ReasoningContractError(ValueError):
    """Raised when a reasoning model response violates the structured contract."""

    def __init__(
        self,
        message: str,
        *,
        prompt_ref: str = "",
        model: str | None = None,
        original_problem: str = "",
        raw_output: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.prompt_ref = prompt_ref
        self.model = model
        self.original_problem = original_problem
        self.raw_output = raw_output


class ReasoningStepSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int
    claim: str
    latex: str = Field(..., description="Raw LaTeX expression for this step, without prose.")
    justification: str


class ReasoningAnswerSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    latex: str = Field(..., description="Raw LaTeX form of the final answer.")


class ReasoningResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    given: str
    steps: List[ReasoningStepSchema]
    answer: ReasoningAnswerSchema


class ReasoningPipeline:
    def __init__(self, manager: ModelManager):
        self.model_manager = manager

    def _task_prompt_ref(self, task: str, default: str) -> str:
        config = getattr(self.model_manager, "config", {}) or {}
        return (
            config.get("tasks", {})
            .get(task, {})
            .get("prompt_ref")
            or default
        )

    def process(self, reasoning_input: ReasoningInput) -> ReasoningOutput:
        variables = {"problem_text": reasoning_input.problem_statement}
        if reasoning_input.visual_context and reasoning_input.visual_context.strip():
            variables["visual_context"] = reasoning_input.visual_context

        prompt_ref = self._task_prompt_ref("reasoning", "reasoning/solve@v2")
        schema = self.schema_for_task_prompt("reasoning", prompt_ref)
        response = self.model_manager.call(
            task="reasoning",
            prompt_ref=prompt_ref,
            variables=variables,
            schema=schema,
        )

        return self.parse_model_response(
            response=response,
            original_problem=reasoning_input.problem_statement,
            prompt_ref=prompt_ref,
            schema=schema,
        )

    def schema_for_task_prompt(self, task: str, prompt_ref: str):
        config = getattr(self.model_manager, "config", {}) or {}
        task_cfg = config.get("tasks", {}).get(task, {})
        return (
            ReasoningResponseSchema
            if task_cfg.get("provider") == "openai"
            and prompt_ref in {"reasoning/solve@v3", "reasoning/repair@v2"}
            else None
        )

    def parse_model_response(
        self,
        response: Any,
        original_problem: str,
        prompt_ref: str,
        schema: Any = None,
    ) -> ReasoningOutput:
        if schema is not None:
            if isinstance(response.parsed, ReasoningResponseSchema):
                return self._from_schema_response(
                    response.parsed,
                    original_problem,
                    response,
                    prompt_ref,
                )
            validation_error = response.meta.get("validation_error") if hasattr(response, "meta") else None
            message = "Reasoning JSON schema response failed validation."
            if validation_error:
                message = f"{message} {validation_error}"
            self._raise_contract_error(
                message,
                response.content,
                original_problem,
                response,
                prompt_ref,
            )

        return self._parse_structured_response(
            response.content,
            original_problem,
            response,
            prompt_ref=prompt_ref,
        )

    def _from_schema_response(
        self,
        parsed: ReasoningResponseSchema,
        original_problem: str,
        response: Any,
        prompt_ref: str,
    ) -> ReasoningOutput:
        steps = [
            ReasoningStep(
                step_number=step.step_number,
                claim=step.claim.strip(),
                latex_expression=step.latex.strip(),
                justification=step.justification.strip(),
            )
            for step in parsed.steps
        ]
        if not steps:
            self._raise_contract_error(
                "Reasoning JSON schema response contains no steps.",
                response.content,
                original_problem,
                response,
                prompt_ref,
            )

        final_answer = parsed.answer.value.strip()
        final_answer_latex = parsed.answer.latex.strip()
        if not final_answer:
            final_answer = self._extract_final_answer(final_answer_latex)
        if not final_answer:
            self._raise_contract_error(
                "Reasoning JSON schema response answer has no value.",
                response.content,
                original_problem,
                response,
                prompt_ref,
            )

        return ReasoningOutput(
            original_problem=original_problem,
            steps=steps,
            final_answer=final_answer,
            think_reasoning="",
            processing_metadata={
                "model_used": response.meta.get("model") if hasattr(response, "meta") else None,
                "prompt_version": prompt_ref,
                "step_count": len(steps),
                "final_answer_latex": final_answer_latex,
                "raw_response_length": len(response.content),
                "reasoning_contract": "json_schema",
            },
        )

    def _parse_structured_response(
        self,
        content: str,
        original_problem: str,
        response: Any,
        prompt_ref: str = "reasoning/solve@v2",
    ) -> ReasoningOutput:
        think_match = _THINK_RE.search(content)
        think_content = think_match.group(2).strip() if think_match else ""

        solution_match = re.search(r'<solution>(.*?)</solution>', content, re.DOTALL)
        if not solution_match:
            self._raise_contract_error(
                "Reasoning response missing required <solution> block.",
                content,
                original_problem,
                response,
                prompt_ref,
            )

        solution_text = solution_match.group(1)

        step_pattern = re.compile(
            r'<step number="(\d+)">\s*'
            r'<claim>(.*?)</claim>\s*'
            r'<latex>(.*?)</latex>\s*'
            r'<justification>(.*?)</justification>\s*'
            r'</step>',
            re.DOTALL
        )
        steps: List[ReasoningStep] = []
        for m in step_pattern.finditer(solution_text):
            steps.append(ReasoningStep(
                step_number=int(m.group(1)),
                claim=m.group(2).strip(),
                latex_expression=m.group(3).strip(),
                justification=m.group(4).strip()
            ))

        answer_match = re.search(
            r'<answer>\s*<value>(.*?)</value>\s*<latex>(.*?)</latex>\s*</answer>',
            solution_text, re.DOTALL
        )
        if not answer_match:
            self._raise_contract_error(
                "Reasoning response missing required <answer> block.",
                content,
                original_problem,
                response,
                prompt_ref,
            )

        final_answer = answer_match.group(1).strip()
        final_answer_latex = answer_match.group(2).strip() if answer_match else ""
        if not final_answer:
            final_answer = self._extract_final_answer(final_answer_latex)
        if not final_answer:
            self._raise_contract_error(
                "Reasoning response answer has no value.",
                content,
                original_problem,
                response,
                prompt_ref,
            )

        if not steps:
            self._raise_contract_error(
                "Reasoning response contains no valid structured steps.",
                content,
                original_problem,
                response,
                prompt_ref,
            )

        return ReasoningOutput(
            original_problem=original_problem,
            steps=steps,
            final_answer=final_answer,
            think_reasoning=think_content,
            processing_metadata={
                "model_used": response.meta.get("model") if hasattr(response, "meta") else None,
                "prompt_version": prompt_ref,
                "step_count": len(steps),
                "final_answer_latex": final_answer_latex,
                "raw_response_length": len(content)
            }
        )

    def _fallback_parse(self, content: str, original_problem: str, response: Any) -> ReasoningOutput:
        raise ReasoningContractError("Unstructured reasoning fallback is disabled.")

    def _raise_contract_error(
        self,
        message: str,
        content: str,
        original_problem: str,
        response: Any,
        prompt_ref: str,
    ) -> None:
        model = response.meta.get("model") if hasattr(response, "meta") else None
        header = (
            "\n=== REASONING CONTRACT FAILURE ===\n"
            f"error: {message}\n"
            f"prompt_ref: {prompt_ref}\n"
            f"model: {model}\n"
            f"problem: {original_problem}\n"
            "=== RAW REASONING MODEL OUTPUT BEGIN ==="
        )
        footer = "=== RAW REASONING MODEL OUTPUT END ==="
        logger.error("%s\n%s\n%s", header, content, footer)
        print(
            "Reasoning contract failure: "
            f"{message} (prompt_ref={prompt_ref}, model={model})"
        )
        raise ReasoningContractError(
            message,
            prompt_ref=prompt_ref,
            model=model,
            original_problem=original_problem,
            raw_output=content,
        )

    def _extract_final_answer(self, text: str) -> str:
        start = text.find('\\boxed{')
        if start >= 0:
            i = start + 7
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                return text[start + 7:i - 1]
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else ""
