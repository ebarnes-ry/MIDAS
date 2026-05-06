from typing import Dict, Any, List, Optional
import json

from .verification_types import VerificationResult, VerificationError, ErrorType
from src.pipeline.reasoning.feedback import FeedbackGenerator
from .codegen import SymPyCodeGenerator
from .executor import SafeExecutor
from .parser import VerificationOutputParser
from ..reasoning.types import ReasoningOutput
from src.models.manager import ModelManager

class VerificationPipeline:
    """
    Implements the "Verification Contract" pipeline.
    This design is deterministic and avoids guessing the root cause of failures.
    """
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.task_config = model_manager.config["tasks"]["verification"]
        self.repair_temperature = self.task_config.get("repair_temperature", 0.1)
        execution_timeout = self.task_config.get("execution_timeout", 30)
        memory_limit_mb = self.task_config.get("memory_limit_mb", 512)

        self.code_generator = SymPyCodeGenerator(model_manager)
        self.executor = SafeExecutor(timeout=execution_timeout, max_memory_mb=memory_limit_mb)
        self.output_parser = VerificationOutputParser()

    def verify(self, reasoning: ReasoningOutput) -> VerificationResult:
        """
        Main verification logic. Follows a Generate -> Execute -> Analyze flow.
        """
        # --- 1. GENERATE INITIAL CODE ---
        try:
            code, metadata = self.code_generator.generate(reasoning)
        except Exception as e:
            return self._create_failure_result(reasoning, f"Initial code generation failed: {e}", generated_code="")

        # --- 2. EXECUTE THE CODE ---
        execution_result = self.executor.execute(code)
        
        # --- 3. ANALYZE THE RESULT ---
        if not execution_result.success:
            # Execution crashed (SyntaxError, RuntimeError, Timeout). This is a CODEGEN FAULT.
            print("Execution failed. Diagnosed as CODEGEN FAULT. Attempting repair...")
            return self._handle_codegen_fault(code, execution_result, reasoning)

        # --- 4. PARSE THE OUTPUT (CONTRACT ADHERENCE) ---
        steps, final_verdict, parsing_error = self.output_parser.parse(execution_result)
        
        if parsing_error:
            # Output did not adhere to the JSON contract. This is a CODEGEN FAULT.
            print(f"Parsing failed due to contract violation: {parsing_error}. Diagnosed as CODEGEN FAULT. Attempting repair...")
            return self._handle_codegen_fault(code, execution_result, reasoning, f"Output parsing failed: {parsing_error}")

        if not final_verdict:
            # Contract violation: missing the final verdict JSON. This is a CODEGEN FAULT.
            print("Missing final verdict. Diagnosed as CODEGEN FAULT. Attempting repair...")
            return self._handle_codegen_fault(code, execution_result, reasoning, "Missing final_answer_verified JSON object in output.")

        # --- 5. CHECK VERIFICATION RESULTS ---
        all_steps_ok = all(s.verified for s in steps)
        answer_ok = final_verdict.get("final_answer_verified", False)

        if all_steps_ok and answer_ok:
            # Everything passed. This is a success.
            print("Verification successful.")
            return self._create_final_result(reasoning, code, execution_result, steps, final_verdict, status="verified")
        else:
            # Code ran but proved the math wrong. This is a REASONING FAULT.
            print("Verification failed. Diagnosed as REASONING FAULT.")
            return self._create_final_result(reasoning, code, execution_result, steps, final_verdict, status="failed_reasoning")

    def _handle_codegen_fault(self, original_code: str, exec_result: Any, reasoning: ReasoningOutput, extra_error: str = None) -> VerificationResult:
        """
        Attempts a single, targeted repair of faulty code generation.
        """
        error_message = exec_result.stderr or extra_error or "Unknown execution error."
        
        # Create a specific repair prompt
        repair_prompt = self._create_codegen_repair_prompt(original_code, error_message)
        
        try:
            # Call LLM for repair
            repaired_code = self._get_repaired_code(reasoning, repair_prompt)
            
            # Re-execute the repaired code
            new_exec_result = self.executor.execute(repaired_code)
            
            # If it still fails, we give up.
            if not new_exec_result.success:
                raise RuntimeError("Repaired code also failed to execute.")
            
            # Analyze the output of the *repaired* code
            steps, final_verdict, parsing_error = self.output_parser.parse(new_exec_result)

            if parsing_error or not final_verdict:
                 raise RuntimeError("Repaired code still violates the verification contract.")
            
            # Check the logic of the now-working code
            status = "verified" if all(s.verified for s in steps) and final_verdict.get("final_answer_verified") else "failed_reasoning"
            return self._create_final_result(reasoning, repaired_code, new_exec_result, steps, final_verdict, status, repaired_from_codegen_fault=True)

        except Exception as e:
            return self._create_failure_result(reasoning, f"Codegen fault repair failed: {e}", generated_code=original_code, errors=[
                VerificationError(error_type=ErrorType.SYNTAX_ERROR, message=error_message)
            ])

    def _create_codegen_repair_prompt(self, code: str, error: str) -> str:
        return f"""The following Python code failed to execute or violated the verification contract.
Error:
---
{error}
---

Original Code:
---
{code}
---
The code MUST adhere to the contract (printing step-by-step JSON results and a final JSON verdict).
Fix the Python code so it is syntactically correct and strictly follows the contract. Do not change the underlying mathematical logic."""

    def _get_repaired_code(self, reasoning: ReasoningOutput, repair_prompt_user_content: str) -> str:
        """Calls the LLM with a specific repair prompt."""
        # We reuse the system prompt from the v3 contract to remind the model of the rules.
        system_prompt = self.model_manager.prompts.load_prompt("codegen/baseline_codegen@v6").system_template
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_prompt_user_content}
        ]
        
        response = self.model_manager.call(
            task="verification",
            prompt_ref="codegen/baseline_codegen@v6", # Use same task config
            variables={'reasoning': reasoning}, # For model/provider context
            messages_override=messages,
            temperature=self.repair_temperature
        )
        
        repaired_code = self.code_generator.extract_code(response.content)
        if not repaired_code:
            raise ValueError("Repair attempt failed to generate any code.")
        return repaired_code
    
    def _annotate_reasoning_steps(
        self,
        reasoning: ReasoningOutput,
        step_verifications: List
    ) -> None:
        """Write verification results back onto ReasoningStep objects in-place."""
        step_map = {sv.step_number: sv for sv in step_verifications}
        for step in reasoning.steps:
            result = step_map.get(step.step_number)
            if result is not None:
                step.verification_status = result.verified
                step.verification_note = result.note if not result.verified else None

    def _create_final_result(self, reasoning, code, exec_result, steps, final_verdict, status, repaired_from_codegen_fault=False) -> VerificationResult:
        """Helper to construct the final VerificationResult object."""
        self._annotate_reasoning_steps(reasoning, steps)
        FeedbackGenerator(self.model_manager).annotate_failed_steps(
            reasoning.original_problem, reasoning.steps
        )
        confidence = self._calculate_confidence(exec_result, steps, final_verdict)
        errors = []
        
        if status == "failed_reasoning":
            failed_steps = [s for s in steps if not s.verified]
            if failed_steps:
                 errors.append(VerificationError(error_type=ErrorType.ASSERTION_FAILED, message=f"Step {failed_steps[0].step_number} failed verification: {failed_steps[0].description}"))
            if not final_verdict.get("final_answer_verified"):
                errors.append(VerificationError(error_type=ErrorType.ANSWER_MISMATCH, message=f"Final answer mismatch. Computed: {final_verdict.get('computed')}, Claimed: {final_verdict.get('claimed')}"))

        return VerificationResult(
            status=status,
            confidence_score=confidence,
            reasoning_output=reasoning,
            generated_code=code,
            execution_result=exec_result,
            step_verifications=steps,
            answer_match=final_verdict.get("final_answer_verified"),
            errors=errors,
            metadata={"repaired_from_codegen_fault": repaired_from_codegen_fault}
        )
        
    def _create_failure_result(self, reasoning, error_msg, generated_code, errors=None) -> VerificationResult:
        """Creates a result for an unrecoverable pipeline failure."""
        return VerificationResult(
            status="failed_pipeline",
            confidence_score=0.0,
            reasoning_output=reasoning,
            generated_code=generated_code,
            errors=errors or [VerificationError(error_type=ErrorType.RUNTIME_ERROR, message=error_msg)],
            metadata={"pipeline_failure": True}
        )

    def _calculate_confidence(self, exec_res, steps, final_verdict) -> float:
        """Calculates a confidence score based on the verification results."""
        if not exec_res.success or not final_verdict:
            return 0.0
        
        score = 0.5 # Base score for successful execution and parsing
        
        if steps:
            step_ratio = sum(1 for s in steps if s.verified) / len(steps)
            score += step_ratio * 0.25
        else: # No steps, but ran
            score += 0.25
            
        if final_verdict.get("final_answer_verified", False):
            score += 0.25
            
        return round(max(0.0, min(1.0, score)), 4)