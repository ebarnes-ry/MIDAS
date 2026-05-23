"""
Verification orchestrator that handles the complete verification pipeline
including reasoning repair when verification fails due to reasoning issues.
"""

import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from .verification import VerificationPipeline
from .verification_types import CodeExecutionResult, ErrorType, VerificationError, VerificationResult
from ..reasoning.reasoning import ReasoningContractError, ReasoningPipeline
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
        self._last_repair_contract_error: Optional[ReasoningContractError] = None

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
                self._last_repair_contract_error = None
                repaired_reasoning = self._attempt_reasoning_repair(current_reasoning, verification_result)
                processing_time = time.time() - start_time

                contract_error = self._last_repair_contract_error
                repair_history.append(RepairAttempt(
                    attempt_number=attempt,
                    repair_type="reasoning",
                    reason=f"Reasoning verification failed with status: {verification_result.status}",
                    success=repaired_reasoning is not None,
                    processing_time=processing_time,
                    error_message=contract_error.message if contract_error else None,
                    repaired_reasoning=repaired_reasoning,
                ))

                if not repaired_reasoning:
                    if contract_error:
                        verification_result = self._create_contract_failure_result(
                            current_reasoning,
                            contract_error,
                            source="reasoning_repair",
                        )
                        self.logger.log_attempt(
                            trajectory_id=tid,
                            attempt_number=attempt + 1,
                            reasoning_output=current_reasoning,
                            verification_result=verification_result,
                            generated_code="",
                        )
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
        except ReasoningContractError as e:
            self._last_repair_contract_error = e
            print(f"Reasoning repair violated the structured contract: {e.message}")
            return None
        except Exception as e:
            print(f"Error during reasoning repair call: {e}")
            return None

    def _create_reasoning_repair_context(self, verification_result: VerificationResult) -> str:
        context_parts = [f"- {error.message}" for error in verification_result.errors]
        failed_steps = [s for s in verification_result.step_verifications if not s.verified]
        result_reasoning = getattr(verification_result, "reasoning_output", None)
        raw_reasoning_steps = getattr(result_reasoning, "steps", [])
        if not isinstance(raw_reasoning_steps, list):
            raw_reasoning_steps = []
        reasoning_steps = {step.step_number: step for step in raw_reasoning_steps}
        if failed_steps:
            context_parts.append("\nThe following steps were proven incorrect:")
            for step in failed_steps:
                reasoning_step = reasoning_steps.get(step.step_number)
                claim = getattr(reasoning_step, "claim", None) or step.description
                context_parts.append(f"  - Step {step.step_number}: {claim}")
                if step.note:
                    context_parts.append(f"    Verifier note: {step.note}")

        metadata = getattr(verification_result, "metadata", {}) or {}
        final_verdict = metadata.get("final_verdict", {})
        if final_verdict and final_verdict.get("final_answer_verified") is False:
            context_parts.append("\nThe final answer was proven incorrect:")
            if final_verdict.get("answer"):
                context_parts.append(f"  - Claimed answer: {final_verdict['answer']}")
            if final_verdict.get("note"):
                context_parts.append(f"  - Verifier note: {final_verdict['note']}")
            elif final_verdict.get("computed") is not None or final_verdict.get("claimed") is not None:
                context_parts.append(f"  - Computed: {final_verdict.get('computed')}")
                context_parts.append(f"  - Claimed: {final_verdict.get('claimed')}")

        return "\n".join(context_parts)

    def _create_contract_failure_result(
        self,
        reasoning: ReasoningOutput,
        error: ReasoningContractError,
        *,
        source: str,
    ) -> VerificationResult:
        return VerificationResult(
            status="failed_contract",
            confidence_score=0.0,
            reasoning_output=reasoning,
            generated_code="",
            execution_result=CodeExecutionResult(
                success=False,
                stdout="",
                stderr=error.message,
                execution_time=0.0,
                exception_type="ReasoningContractError",
                exception_message=error.message,
                exception_traceback="",
            ),
            answer_match=None,
            errors=[
                VerificationError(
                    error_type=ErrorType.CONTRACT_VIOLATION,
                    message=error.message,
                )
            ],
            metadata={
                "contract_failure": True,
                "contract_source": source,
                "prompt_ref": error.prompt_ref,
                "model": error.model,
                "raw_response_length": len(error.raw_output or ""),
            },
        )
