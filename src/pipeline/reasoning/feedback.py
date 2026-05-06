from typing import List
from src.models.manager import ModelManager
from .types import ReasoningStep


class FeedbackGenerator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_step_feedback(self, problem_statement: str, step: ReasoningStep) -> str:
        """
        Generate targeted feedback for a single failed step.
        Returns an empty string if called on a passing or unchecked step.
        """
        if step.verification_status is not False:
            return ""

        try:
            response = self.model_manager.call(
                task="reasoning",
                prompt_ref="reasoning/student_feedback@v1",
                variables={
                    "problem_statement": problem_statement,
                    "step_number": step.step_number,
                    "claim": step.claim,
                    "latex_expression": step.latex_expression or "",
                    "justification": step.justification,
                    "verification_note": step.verification_note or "Verification failed — no detail available.",
                }
            )
            return response.content.strip()
        except Exception as e:
            return f"Could not generate feedback: {e}"

    def annotate_failed_steps(self, problem_statement: str, steps: List[ReasoningStep]) -> None:
        """
        Generate feedback for all failed steps and write it back in-place.
        Skips steps that passed or haven't been checked.
        """
        for step in steps:
            if step.verification_status is False:
                step.feedback = self.generate_step_feedback(problem_statement, step)
