from typing import Dict, Any, List, Optional, Tuple
import json
import re

from .verification_types import (
    CodeExecutionResult,
    ErrorType,
    VerificationError,
    VerificationResult,
    VerificationStatus,
)
from src.pipeline.reasoning.feedback import FeedbackGenerator
from .codegen import SymPyCodeGenerator, CodegenContractError
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
        boundary_result = self._classify_verification_boundary(reasoning)
        if boundary_result is not None:
            return boundary_result

        # --- 1. GENERATE INITIAL CODE ---
        try:
            code, metadata = self.code_generator.generate(reasoning)
        except CodegenContractError as e:
            fault_status = (
                VerificationStatus.FAILED_CODEGEN
                if e.category == "syntax"
                else VerificationStatus.FAILED_CONTRACT
            )
            fault_error_type = (
                ErrorType.SYNTAX_ERROR
                if e.category == "syntax"
                else ErrorType.CONTRACT_VIOLATION
            )
            exec_result = CodeExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                exception_type="CodegenContractError",
                exception_message=str(e),
                exception_traceback=str(e),
            )
            print("Generated code violated the codegen contract. Attempting repair...")
            result = self._handle_codegen_fault(
                e.code,
                exec_result,
                reasoning,
                fault_status=fault_status,
                fault_error_type=fault_error_type,
                fault_category=e.category,
            )
            return self._apply_unsupported_boundary_after_failure(reasoning, result)
        except Exception as e:
            result = self._create_failure_result(
                reasoning,
                f"Initial code generation failed: {e}",
                generated_code="",
                status=VerificationStatus.FAILED_CODEGEN,
                error_type=ErrorType.RUNTIME_ERROR,
                metadata={"codegen_failure": True},
            )
            return self._apply_unsupported_boundary_after_failure(reasoning, result)

        # --- 2. EXECUTE THE CODE ---
        execution_result = self.executor.execute(code)
        
        # --- 3. ANALYZE THE RESULT ---
        if not execution_result.success:
            # Execution crashed (SyntaxError, RuntimeError, Timeout). This is a CODEGEN FAULT.
            if self._is_unsupported_error(execution_result.stderr, execution_result):
                print("Execution failed due to unsupported symbolic operation.")
                return self._create_failure_result(
                    reasoning,
                    self._execution_error_message(execution_result),
                    generated_code=code,
                    status=VerificationStatus.UNSUPPORTED,
                    error_type=ErrorType.SYMBOLIC_FAILURE,
                    metadata={"unsupported": True, "unsupported_source": "sympy_execution"},
                    execution_result=execution_result,
                )
            print("Execution failed. Diagnosed as CODEGEN FAULT. Attempting repair...")
            result = self._handle_codegen_fault(
                code,
                execution_result,
                reasoning,
                fault_status=VerificationStatus.FAILED_CODEGEN,
                fault_error_type=self._error_type_for_execution(execution_result),
                fault_category=self._fault_category_for_execution(execution_result),
            )
            return self._apply_unsupported_boundary_after_failure(reasoning, result)

        # --- 4. PARSE THE OUTPUT (CONTRACT ADHERENCE) ---
        expected_step_numbers = self._expected_step_numbers(reasoning)
        steps, final_verdict, parsing_error = self.output_parser.parse(
            execution_result,
            expected_step_numbers=expected_step_numbers,
        )
        
        if parsing_error:
            # Output did not adhere to the JSON contract. This is a CODEGEN FAULT.
            print(f"Parsing failed due to contract violation: {parsing_error}. Attempting repair...")
            result = self._handle_codegen_fault(
                code,
                execution_result,
                reasoning,
                f"Output parsing failed: {parsing_error}",
                fault_status=VerificationStatus.FAILED_CONTRACT,
                fault_error_type=ErrorType.CONTRACT_VIOLATION,
                fault_category="output_contract",
            )
            return self._apply_unsupported_boundary_after_failure(reasoning, result)

        if not final_verdict:
            # Contract violation: missing the final verdict JSON. This is a CODEGEN FAULT.
            print("Missing final verdict. Attempting contract repair...")
            result = self._handle_codegen_fault(
                code,
                execution_result,
                reasoning,
                "Missing final_answer_verified JSON object in output.",
                fault_status=VerificationStatus.FAILED_CONTRACT,
                fault_error_type=ErrorType.CONTRACT_VIOLATION,
                fault_category="missing_final_verdict",
            )
            return self._apply_unsupported_boundary_after_failure(reasoning, result)

        # --- 5. CHECK VERIFICATION RESULTS ---
        all_steps_ok = all(s.verified for s in steps)
        answer_ok = final_verdict.get("final_answer_verified", False)

        if all_steps_ok and answer_ok:
            # Everything passed. This is a success.
            print("Verification successful.")
            return self._create_final_result(reasoning, code, execution_result, steps, final_verdict, status=VerificationStatus.VERIFIED)
        else:
            # Code ran but proved the math wrong. This is a REASONING FAULT.
            print("Verification failed. Diagnosed as REASONING FAULT.")
            return self._create_final_result(reasoning, code, execution_result, steps, final_verdict, status=VerificationStatus.FAILED_REASONING)

    def _handle_codegen_fault(
        self,
        original_code: str,
        exec_result: Any,
        reasoning: ReasoningOutput,
        extra_error: str = None,
        *,
        fault_status: VerificationStatus = VerificationStatus.FAILED_CODEGEN,
        fault_error_type: ErrorType = ErrorType.RUNTIME_ERROR,
        fault_category: str = "runtime",
    ) -> VerificationResult:
        """
        Attempts a single, targeted repair of faulty code generation.
        """
        error_message = extra_error or self._execution_error_message(exec_result)
        
        # Create a specific repair prompt
        repair_prompt = self._create_codegen_repair_prompt(original_code, error_message, fault_category)
        
        try:
            # Call LLM for repair
            repaired_code = self._get_repaired_code(reasoning, repair_prompt)
            
            # Re-execute the repaired code
            new_exec_result = self.executor.execute(repaired_code)
            
            # If it still fails, we give up.
            if not new_exec_result.success:
                raise RuntimeError("Repaired code also failed to execute.")
            
            # Analyze the output of the *repaired* code
            steps, final_verdict, parsing_error = self.output_parser.parse(
                new_exec_result,
                expected_step_numbers=self._expected_step_numbers(reasoning),
            )

            if parsing_error or not final_verdict:
                 raise RuntimeError("Repaired code still violates the verification contract.")
            
            # Check the logic of the now-working code
            status = VerificationStatus.VERIFIED if all(s.verified for s in steps) and final_verdict.get("final_answer_verified") else VerificationStatus.FAILED_REASONING
            return self._create_final_result(reasoning, repaired_code, new_exec_result, steps, final_verdict, status, repaired_from_codegen_fault=True)

        except Exception as e:
            if self._is_unsupported_error(error_message, exec_result) or self._is_unsupported_error(str(e)):
                return self._create_failure_result(
                    reasoning,
                    f"Unsupported verification target: {e}",
                    generated_code=original_code,
                    status=VerificationStatus.UNSUPPORTED,
                    error_type=ErrorType.SYMBOLIC_FAILURE,
                    metadata={"unsupported": True, "unsupported_source": "codegen_repair"},
                    execution_result=exec_result if isinstance(exec_result, CodeExecutionResult) else None,
                )
            return self._create_failure_result(
                reasoning,
                f"Codegen fault repair failed: {e}",
                generated_code=original_code,
                errors=[
                    VerificationError(error_type=fault_error_type, message=error_message)
                ],
                status=fault_status,
                metadata={
                    "codegen_failure": fault_status == VerificationStatus.FAILED_CODEGEN,
                    "contract_failure": fault_status == VerificationStatus.FAILED_CONTRACT,
                    "codegen_fault_category": fault_category,
                },
                execution_result=exec_result if isinstance(exec_result, CodeExecutionResult) else None,
            )

    def _codegen_repair_instructions_for_category(self, category: str) -> str:
        instructions = {
            "emit_final_count": (
                "Category-specific repair for emit_final_count:\n"
                "- Keep every existing emit_step(...) call and its mathematical verified value.\n"
                "- Remove every extra emit_final(...) call.\n"
                "- If there is no emit_final(...) call, add exactly one final emit_final(...) after all step checks.\n"
                "- The single emit_final(...) must be the last verification emission in the script.\n"
                "- Do not put emit_final(...) inside both try and except branches; that still counts as multiple calls and violates the static contract.\n"
                "- If final-answer checking needs error handling, compute final_verified, final_answer, and final_note variables inside the try/except, then call emit_final(final_verified, final_answer, final_note) exactly once after the try/except.\n"
                "- Do not change mathematical checks, symbolic expressions, or step booleans except where needed to route values through the required helpers."
            ),
            "missing_helper": (
                "Category-specific repair for missing_helper:\n"
                "- Add the required emit_step(...) and/or emit_final(...) helper definitions using the v7 contract shape.\n"
                "- Route every step emission through emit_step(...) and the final verdict through exactly one emit_final(...).\n"
                "- Do not change the mathematical checks."
            ),
            "forbidden_json_dumps": (
                "Category-specific repair for forbidden_json_dumps:\n"
                "- Remove all direct print(json.dumps(...)) calls outside emit_step/emit_final.\n"
                "- Convert those emissions to emit_step(...) or the single emit_final(...), preserving the same verified values and notes.\n"
                "- Do not change the mathematical checks."
            ),
            "simplify_instance_method": (
                "Category-specific repair for simplify_instance_method:\n"
                "- Replace instance calls like (a - b).simplify() with sp.simplify(sp.sympify(a) - sp.sympify(b)).\n"
                "- Prefer the contract helper same_expr(a, b) when comparing symbolic expressions.\n"
                "- Do not change the intended equality being checked."
            ),
            "syntax": (
                "Category-specific repair for syntax:\n"
                "- Fix only Python syntax and string-literal issues.\n"
                "- Rewrite long descriptions/notes as safe single-line strings if needed.\n"
                "- Do not change mathematical checks."
            ),
            "runtime_name_error": (
                "Category-specific repair for runtime_name_error:\n"
                "- Define missing local variables before use or replace incorrect variable names with the already-defined intended names.\n"
                "- If the NameError is for a non-allowed builtin, avoid that builtin and use simpler explicit logic.\n"
                "- Do not change mathematical checks except to use the correctly defined variable."
            ),
            "missing_final_verdict": (
                "Category-specific repair for missing_final_verdict:\n"
                "- Add exactly one emit_final(...) call after all emit_step(...) calls.\n"
                "- Base the final verdict on the existing computed final-answer check if present.\n"
                "- Do not add additional step emissions or change mathematical checks."
            ),
            "output_contract": (
                "Category-specific repair for output_contract:\n"
                "- Ensure emitted JSON keys match the v7 contract exactly.\n"
                "- Use emit_step(...) for each step and exactly one emit_final(...) for the final verdict.\n"
                "- Convert all SymPy values and booleans before JSON serialization."
            ),
        }
        return instructions.get(
            category,
            "Category-specific repair for generic runtime/codegen fault:\n"
            "- Fix the reported failure while preserving the existing mathematical checks.\n"
            "- Keep one emit_step(...) per reasoning step and exactly one emit_final(...).",
        )

    def _create_codegen_repair_prompt(self, code: str, error: str, category: str = "runtime") -> str:
        category_instructions = self._codegen_repair_instructions_for_category(category)
        return f"""The following Python code failed to execute or violated the verification contract.
Fault category: {category}

Error:
---
{error}
---

Original Code:
---
{code}
---
{category_instructions}

The code MUST adhere to the verification contract:
- print exactly one JSON object per step and one final verdict JSON object
- define and use emit_step(...) for every step
- define and use emit_final(...) exactly once for the final verdict
- do not call print(json.dumps(...)) directly outside emit_step/emit_final
- convert SymPy BooleanTrue/BooleanFalse and other SymPy values before JSON serialization
- convert notes, answers, and symbolic objects to str(...) before emitting JSON

Important SymPy rules:
- Do not call `.simplify()` as an instance method on arbitrary expressions.
- Use `sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0` or the contract helper `same_expr(a, b)` for equality checks.
- For equation-solving steps, compare `sp.Eq(...)` objects with `same_equation(...)` so swapped sides like `1 = t` and `t = 1` are treated as equivalent.
- If the prior code failed with "Object of type BooleanTrue is not JSON serializable", the fix is to route every verified value through to_json_bool inside emit_step/emit_final.
- If the prior code failed with a SyntaxError such as an unterminated string literal, rewrite descriptions/notes as safe single-line strings or assign them with repr-safe literals.

Return raw Python only. No markdown fences. Do not change the underlying mathematical logic."""

    def _get_repaired_code(self, reasoning: ReasoningOutput, repair_prompt_user_content: str) -> str:
        """Calls the LLM with a specific repair prompt."""
        prompt_ref = self.task_config.get("prompt_ref", "codegen/baseline_codegen@v7")
        system_prompt = self.model_manager.prompts.load_prompt(prompt_ref).system_template
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_prompt_user_content}
        ]
        
        response = self.model_manager.call(
            task="verification",
            prompt_ref=prompt_ref,
            variables={'reasoning': reasoning}, # For model/provider context
            messages_override=messages,
            temperature=self.repair_temperature
        )
        
        repaired_code = self.code_generator.extract_code(response.content)
        if not repaired_code:
            raise ValueError("Repair attempt failed to generate any code.")
        self.code_generator.validate_code_contract(repaired_code)
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

    def _expected_step_numbers(self, reasoning: ReasoningOutput) -> List[int]:
        return [int(step.step_number) for step in reasoning.steps]

    def _final_answer_mismatch_message(self, final_verdict: Dict[str, Any]) -> str:
        answer = final_verdict.get("answer")
        note = final_verdict.get("note")
        computed = final_verdict.get("computed")
        claimed = final_verdict.get("claimed")

        if computed is not None or claimed is not None:
            return f"Final answer mismatch. Computed: {computed}, Claimed: {claimed}"

        parts = []
        if answer not in (None, ""):
            parts.append(f"Answer: {answer}")
        if note not in (None, ""):
            parts.append(f"Verifier note: {note}")
        if parts:
            return "Final answer mismatch. " + "; ".join(parts)
        return "Final answer mismatch."

    def _create_final_result(self, reasoning, code, exec_result, steps, final_verdict, status, repaired_from_codegen_fault=False) -> VerificationResult:
        """Helper to construct the final VerificationResult object."""
        self._annotate_reasoning_steps(reasoning, steps)
        FeedbackGenerator(self.model_manager).annotate_failed_steps(
            reasoning.original_problem, reasoning.steps
        )
        confidence = self._calculate_confidence(exec_result, steps, final_verdict)
        errors = []
        
        if status == VerificationStatus.FAILED_REASONING:
            failed_steps = [s for s in steps if not s.verified]
            if failed_steps:
                 errors.append(VerificationError(error_type=ErrorType.ASSERTION_FAILED, message=f"Step {failed_steps[0].step_number} failed verification: {failed_steps[0].description}"))
            if not final_verdict.get("final_answer_verified"):
                errors.append(VerificationError(error_type=ErrorType.ANSWER_MISMATCH, message=self._final_answer_mismatch_message(final_verdict)))

        return VerificationResult(
            status=status,
            confidence_score=confidence,
            reasoning_output=reasoning,
            generated_code=code,
            execution_result=exec_result,
            step_verifications=steps,
            answer_match=final_verdict.get("final_answer_verified"),
            errors=errors,
            metadata={
                "repaired_from_codegen_fault": repaired_from_codegen_fault,
                "final_verdict": final_verdict,
            }
        )
        
    def _create_failure_result(
        self,
        reasoning,
        error_msg,
        generated_code,
        errors=None,
        *,
        status: VerificationStatus = VerificationStatus.FAILED_PIPELINE,
        error_type: ErrorType = ErrorType.RUNTIME_ERROR,
        metadata: Optional[Dict[str, Any]] = None,
        execution_result: Optional[CodeExecutionResult] = None,
    ) -> VerificationResult:
        """Creates a result for an unrecoverable pipeline failure."""
        return VerificationResult(
            status=status,
            confidence_score=0.0,
            reasoning_output=reasoning,
            generated_code=generated_code,
            execution_result=execution_result,
            errors=errors or [VerificationError(error_type=error_type, message=error_msg)],
            metadata=metadata or {"pipeline_failure": True}
        )

    def _classify_verification_boundary(
        self,
        reasoning: ReasoningOutput,
    ) -> Optional[VerificationResult]:
        metadata = getattr(reasoning, "processing_metadata", {}) or {}
        problem_type = str(metadata.get("problem_type") or "unknown").lower()
        problem_text = f"{reasoning.original_problem}\n{reasoning.worked_solution}".lower()

        visual_required = bool(metadata.get("visual_context_required"))
        visual_attached = bool(metadata.get("visual_context_attached"))
        if visual_required and not visual_attached:
            return self._create_failure_result(
                reasoning,
                "This problem appears to require a diagram, graph, table, or figure, but no usable visual context was attached.",
                generated_code="",
                status=VerificationStatus.NEEDS_VISUAL_CONTEXT,
                error_type=ErrorType.SYMBOLIC_FAILURE,
                metadata={
                    "needs_visual_context": True,
                    "visual_context_required": True,
                    "visual_context_attached": False,
                    "unsupported_reason": "missing_visual_context",
                    "problem_type": problem_type,
                    "verification_boundary": "missing_visual_context",
                },
            )

        return None

    def _unsupported_boundary_reason(self, reasoning: ReasoningOutput) -> Optional[Tuple[str, str]]:
        metadata = getattr(reasoning, "processing_metadata", {}) or {}
        problem_type = str(metadata.get("problem_type") or "unknown").lower()
        problem_text = f"{reasoning.original_problem}\n{reasoning.worked_solution}".lower()

        if self._is_abstract_proof_boundary(problem_type, problem_text):
            return (
                "abstract_proof_verification_boundary",
                "This solution is proof-oriented, and the current SymPy verifier cannot reliably check abstract proof obligations.",
            )

        if self._is_geometry_boundary(problem_type, problem_text):
            return (
                "geometry_symbolic_verification_boundary",
                "This problem is geometry-oriented; the current SymPy verifier cannot reliably validate diagram/theorem-based geometry reasoning.",
            )

        if self._is_advanced_analysis_boundary(problem_type, problem_text):
            return (
                "advanced_analysis_verification_boundary",
                "This problem appears to require advanced analysis reasoning beyond the current symbolic verifier boundary.",
            )

        return None

    def _apply_unsupported_boundary_after_failure(
        self,
        reasoning: ReasoningOutput,
        result: VerificationResult,
    ) -> VerificationResult:
        if result.status in {
            VerificationStatus.VERIFIED,
            VerificationStatus.FAILED_REASONING,
            VerificationStatus.NEEDS_VISUAL_CONTEXT,
            VerificationStatus.UNSUPPORTED,
        }:
            return result

        boundary = self._unsupported_boundary_reason(reasoning)
        if boundary is None:
            return result

        reason, message = boundary
        metadata = getattr(reasoning, "processing_metadata", {}) or {}
        underlying_metadata = dict(result.metadata or {})
        return self._create_failure_result(
            reasoning,
            message,
            generated_code=result.generated_code,
            status=VerificationStatus.UNSUPPORTED,
            error_type=ErrorType.SYMBOLIC_FAILURE,
            metadata={
                **underlying_metadata,
                "unsupported": True,
                "unsupported_source": "post_verification_failure_boundary",
                "unsupported_reason": reason,
                "problem_type": str(metadata.get("problem_type") or "unknown").lower(),
                "verification_boundary": reason,
                "underlying_verification_status": result.status,
                "underlying_error_type": result.errors[0].error_type.value if result.errors else None,
                "visual_context_required": bool(metadata.get("visual_context_required")),
                "visual_context_attached": bool(metadata.get("visual_context_attached")),
            },
            execution_result=result.execution_result,
        )

    def _create_unsupported_boundary_result(
        self,
        reasoning: ReasoningOutput,
        problem_type: str,
        reason: str,
        message: str,
    ) -> VerificationResult:
        metadata = getattr(reasoning, "processing_metadata", {}) or {}
        return self._create_failure_result(
            reasoning,
            message,
            generated_code="",
            status=VerificationStatus.UNSUPPORTED,
            error_type=ErrorType.SYMBOLIC_FAILURE,
            metadata={
                "unsupported": True,
                "unsupported_source": "pre_verification_boundary",
                "unsupported_reason": reason,
                "problem_type": problem_type,
                "verification_boundary": reason,
                "visual_context_required": bool(metadata.get("visual_context_required")),
                "visual_context_attached": bool(metadata.get("visual_context_attached")),
            },
        )

    def _is_abstract_proof_boundary(self, problem_type: str, text: str) -> bool:
        if problem_type == "proof":
            return True
        proof_markers = (
            "prove that",
            "show that",
            "if and only if",
            "necessary and sufficient",
            "for all",
            "there exists",
            "∀",
            "∃",
        )
        return any(marker in text for marker in proof_markers)

    def _is_geometry_boundary(self, problem_type: str, text: str) -> bool:
        if problem_type == "geometry":
            return True
        geometry_markers = (
            "triangle",
            "circle",
            "angle",
            "congruent triangles",
            "similar triangles",
            "parallel lines",
            "perpendicular",
        )
        symbolic_markers = (
            "solve",
            "equation",
            "differentiate",
            "integrate",
            "limit",
            "factor",
            "expand",
        )
        return any(marker in text for marker in geometry_markers) and not any(
            marker in text for marker in symbolic_markers
        )

    def _is_advanced_analysis_boundary(self, problem_type: str, text: str) -> bool:
        advanced_markers = (
            "uniform convergence",
            "pointwise convergence",
            "measure",
            "lebesgue",
            "compact",
            "banach",
            "hilbert",
            "epsilon-delta",
            "epsilon delta",
            "real analysis",
            "complex analysis",
        )
        simple_calculus = re.search(r"\b(derivative|differentiate|integral|integrate|limit)\b", text)
        return any(marker in text for marker in advanced_markers) and not simple_calculus

    def _execution_error_message(self, exec_result: Any) -> str:
        if not exec_result:
            return "Unknown execution error."
        if getattr(exec_result, "stderr", None):
            return exec_result.stderr
        if getattr(exec_result, "exception_message", None):
            return exec_result.exception_message
        return "Unknown execution error."

    def _error_type_for_execution(self, exec_result: CodeExecutionResult) -> ErrorType:
        exception_type = (exec_result.exception_type or "").lower()
        if "syntax" in exception_type:
            return ErrorType.SYNTAX_ERROR
        if "import" in exception_type:
            return ErrorType.IMPORT_ERROR
        if "timeout" in exception_type:
            return ErrorType.TIMEOUT
        return ErrorType.RUNTIME_ERROR

    def _fault_category_for_execution(self, exec_result: CodeExecutionResult) -> str:
        exception_type = (exec_result.exception_type or "").lower()
        if "nameerror" in exception_type:
            return "runtime_name_error"
        if "syntax" in exception_type:
            return "syntax"
        return "runtime"

    def _is_unsupported_error(self, error_text: str, exec_result: Optional[CodeExecutionResult] = None) -> bool:
        text = (
            f"{error_text or ''} "
            f"{getattr(exec_result, 'exception_type', '') or ''} "
            f"{getattr(exec_result, 'exception_message', '') or ''}"
        ).lower()
        unsupported_markers = (
            "notimplementederror",
            "not implemented",
            "not supported",
            "unsupported",
            "sympifyerror",
            "parseexception",
            "could not parse",
            "unable to parse",
            "cannot determine truth value of relational",
            "no algorithms are implemented",
            "multiple generators",
            "solveset is unable",
        )
        return any(marker in text for marker in unsupported_markers)

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
