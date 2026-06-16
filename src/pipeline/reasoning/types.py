from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


@dataclass
class ReasoningStep:
    """
    A single logical step in a mathematical proof or solution.

    Each step has a human-readable claim (what is being asserted),
    an optional LaTeX expression (the symbolic form of the claim),
    and a justification (why this follows from previous steps).

    The `verification_status` and `verification_note` fields are
    left empty at reasoning time and filled in by the verification
    pipeline after SymPy execution.
    """
    step_number: int
    claim: str
    justification: str
    latex_expression: Optional[str] = None
    verification_status: Optional[bool] = None
    verification_note: Optional[str] = None
    feedback: Optional[str] = None


@dataclass
class ReasoningInput:
    problem_statement: str
    visual_context: Optional[str] = None
    source_metadata: Optional[Dict[str, Any]] = None


@dataclass(init=False)
class ReasoningOutput:
    original_problem: str
    steps: List[ReasoningStep]
    final_answer: str
    think_reasoning: str
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        original_problem: str,
        steps: Optional[List[ReasoningStep]] = None,
        final_answer: str = "",
        think_reasoning: str = "",
        processing_metadata: Optional[Dict[str, Any]] = None,
        worked_solution: Optional[str] = None,
    ):
        self.original_problem = original_problem
        if steps is None:
            steps = self._steps_from_legacy_worked_solution(worked_solution)
        self.steps = steps
        self.final_answer = final_answer
        self.think_reasoning = think_reasoning
        self.processing_metadata = processing_metadata or {}

    @staticmethod
    def _steps_from_legacy_worked_solution(worked_solution: Optional[str]) -> List[ReasoningStep]:
        if not worked_solution:
            return []
        return [
            ReasoningStep(
                step_number=1,
                claim=worked_solution,
                justification="legacy worked_solution input",
            )
        ]

    @property
    def worked_solution(self) -> str:
        """
        Backwards-compatible property. Returns a plain-text reconstruction
        of the solution from the structured steps.
        """
        lines = []
        for s in self.steps:
            lines.append(f"{s.step_number}. {s.claim}")
            if s.latex_expression:
                lines.append(f"   {s.latex_expression}")
            lines.append(f"   ({s.justification})")
        return "\n".join(lines)
