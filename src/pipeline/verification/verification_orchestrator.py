"""
Verification orchestrator that handles the complete verification pipeline
including reasoning repair when verification fails due to reasoning issues.
"""

import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from .verification import VerificationPipeline
from .verification_types import VerificationResult
from ..reasoning.reasoning import ReasoningPipeline
from ..reasoning.types import ReasoningOutput
from src.models.manager import ModelManager
from src.pipeline.trajectory import TrajectoryLogger


@dataclass
class RepairAttempt:
    attempt_number: int
    repair_type: str
    reason: str
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    repaired_reasoning: Optional[ReasoningOutput] = field(default=None, repr=False)


class VerificationOrchestrator:
    """
    Orchestrates the full verification pipeline with reasoning repair capability.
    """

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.verification_pipeline = VerificationPipeline(model_manager)
        self.reasoning_pipeline = ReasoningPipeline(model_manager)
        log_path = model_manager.config.get("trajectory_log_path", "trajectories/midas_trajectories.jsonl")
        self.logger = TrajectoryLogger(log_path=log_path)
        demo = model_manager.config.get("demo", {})
        self._max_repair_attempts = demo.get("max_repair_attempts", 2)
        self._max_tokens_per_request = demo.get("max_tokens_per_request", None)

    def verify_with_repair(
        self,
        reasoning_output: ReasoningOutput,
        max_reasoning_attempts: int | None = None,
    ) -> Tuple[VerificationResult, List[RepairAttempt]]:
        # demo config caps override the call-site default
        if max_reasoning_attempts is None:
            max_reasoning_attempts = self._max_repair_attempts
        repair_history: List[RepairAttempt] = []
        current_reasoning = reasoning_output
        verification_result: Optional[VerificationResult] = None

        problem_type = getattr(reasoning_output, "problem_type", "unknown")
        tid = self.logger.start_trajectory(
            problem_statement=reasoning_output.original_problem,
            problem_type=str(problem_type),
        )

        for attempt in range(max_reasoning_attempts + 1):
            if attempt > 0:
                print(f"--- Reasoning Repair Attempt {attempt}/{max_reasoning_attempts} ---")
                start_time = time.time()
                repaired_reasoning = self._attempt_reasoning_repair(current_reasoning, verification_result)
                processing_time = time.time() - start_time

                repair_history.append(RepairAttempt(
                    attempt_number=attempt,
                    repair_type="reasoning",
                    reason=f"Reasoning verification failed with status: {verification_result.status}",
                    success=repaired_reasoning is not None,
                    processing_time=processing_time,
                    repaired_reasoning=repaired_reasoning,
                ))

                if not repaired_reasoning:
                    print("Reasoning repair failed to produce a new solution. Halting.")
                    break

                current_reasoning = repaired_reasoning

            verification_result = self.verification_pipeline.verify(current_reasoning)

            self.logger.log_attempt(
                trajectory_id=tid,
                attempt_number=attempt + 1,
                reasoning_output=current_reasoning,
                verification_result=verification_result,
                generated_code=getattr(verification_result, "generated_code", ""),
            )

            if verification_result.status == "verified":
                print("Verification successful.")
                break

            if verification_result.status != "failed_reasoning":
                print(f"Halting repair loop due to non-reasoning error: {verification_result.status}")
                break

        self.logger.close_trajectory(
            trajectory_id=tid,
            final_status=verification_result.status if verification_result else "failed_pipeline",
            max_attempts=max_reasoning_attempts + 1,
        )

        return verification_result, repair_history

    def _attempt_reasoning_repair(
        self,
        failed_reasoning: ReasoningOutput,
        verification_result: VerificationResult,
    ) -> Optional[ReasoningOutput]:
        try:
            feedback = self._create_reasoning_repair_context(verification_result)
            config = getattr(self.model_manager, "config", {}) or {}
            if not isinstance(config, dict):
                config = {}
            prompt_ref = (
                config.get("tasks", {})
                .get("reasoning_repair", {})
                .get("prompt_ref")
                or "reasoning/repair@v1"
            )
            schema = None
            if prompt_ref == "reasoning/repair@v2" and hasattr(self.reasoning_pipeline, "schema_for_task_prompt"):
                schema = self.reasoning_pipeline.schema_for_task_prompt("reasoning_repair", prompt_ref)
            response = self.model_manager.call(
                task="reasoning_repair",
                prompt_ref=prompt_ref,
                variables={
                    "original_problem": failed_reasoning.original_problem,
                    "failed_solution": failed_reasoning.worked_solution,
                    "verification_feedback": feedback,
                },
                schema=schema,
            )
            if schema is not None and hasattr(self.reasoning_pipeline, "parse_model_response"):
                repaired = self.reasoning_pipeline.parse_model_response(
                    response=response,
                    original_problem=failed_reasoning.original_problem,
                    prompt_ref=prompt_ref,
                    schema=schema,
                )
            else:
                repaired = self.reasoning_pipeline._parse_structured_response(
                    response.content,
                    failed_reasoning.original_problem,
                    response,
                )
            repaired.processing_metadata.update({
                "source": "reasoning_repair",
                "original_failure": verification_result.status,
            })
            return repaired
        except Exception as e:
            print(f"Error during reasoning repair call: {e}")
            return None

    def _create_reasoning_repair_context(self, verification_result: VerificationResult) -> str:
        context_parts = [f"- {error.message}" for error in verification_result.errors]
        failed_steps = [s for s in verification_result.step_verifications if not s.verified]
        if failed_steps:
            context_parts.append("\nThe following steps were proven incorrect:")
            for step in failed_steps:
                context_parts.append(f"  - Step {step.step_number}: {step.description}")
        return "\n".join(context_parts)
