import re
from typing import Dict, Any, List, Optional
from src.models.manager import ModelManager
from .types import ReasoningInput, ReasoningOutput, ReasoningStep

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


class ReasoningPipeline:
    def __init__(self, manager: ModelManager):
        self.model_manager = manager

    def process(self, reasoning_input: ReasoningInput) -> ReasoningOutput:
        variables = {"problem_text": reasoning_input.problem_statement}
        if reasoning_input.visual_context and reasoning_input.visual_context.strip():
            variables["visual_context"] = reasoning_input.visual_context

        response = self.model_manager.call(
            task="reasoning",
            prompt_ref="reasoning/solve@v2",
            variables=variables
        )

        return self._parse_structured_response(
            response.content,
            reasoning_input.problem_statement,
            response
        )

    def _parse_structured_response(
        self,
        content: str,
        original_problem: str,
        response: Any
    ) -> ReasoningOutput:
        think_match = _THINK_RE.search(content)
        think_content = think_match.group(2).strip() if think_match else ""

        solution_match = re.search(r'<solution>(.*?)</solution>', content, re.DOTALL)
        if not solution_match:
            print("WARNING: Model did not produce structured v2 output. Falling back to legacy parser.")
            return self._fallback_parse(content, original_problem, response)

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
        final_answer = answer_match.group(1).strip() if answer_match else ""
        final_answer_latex = answer_match.group(2).strip() if answer_match else ""

        if not steps:
            print("WARNING: No structured steps found in v2 output. Falling back.")
            return self._fallback_parse(content, original_problem, response)

        return ReasoningOutput(
            original_problem=original_problem,
            steps=steps,
            final_answer=final_answer,
            think_reasoning=think_content,
            processing_metadata={
                "model_used": response.meta.get("model") if hasattr(response, "meta") else None,
                "prompt_version": "reasoning/solve@v2",
                "step_count": len(steps),
                "final_answer_latex": final_answer_latex,
                "raw_response_length": len(content)
            }
        )

    def _fallback_parse(self, content: str, original_problem: str, response: Any) -> ReasoningOutput:
        think_match = _THINK_RE.search(content)
        think_content = think_match.group(2).strip() if think_match else ""
        worked = _STRIP_THINK_RE.sub('', content).strip()

        return ReasoningOutput(
            original_problem=original_problem,
            steps=[ReasoningStep(
                step_number=1,
                claim=worked,
                justification="(unstructured response — v1 fallback)",
            )],
            final_answer=self._extract_final_answer(worked),
            think_reasoning=think_content,
            processing_metadata={
                "model_used": response.meta.get("model") if hasattr(response, "meta") else None,
                "prompt_version": "reasoning/solve@v1-fallback",
            }
        )

    def _extract_final_answer(self, text: str) -> str:
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed.group(1)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else ""
