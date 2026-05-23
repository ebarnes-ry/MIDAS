from typing import List, Tuple, Optional, Dict, Any, Iterable
import json
import re
from .verification_types import StepVerification, CodeExecutionResult


class VerificationOutputParser:
    def parse(
        self,
        execution_result: CodeExecutionResult,
        expected_step_numbers: Optional[Iterable[int]] = None,
    ) -> Tuple[List[StepVerification], Optional[Dict], Optional[str]]:
        """
        Parse SymPy execution stdout into structured step verifications.

        Returns:
            (step_verifications, final_verdict_dict, parsing_error_string)
            parsing_error_string is None on success.
        """
        steps: List[StepVerification] = []
        final_verdict: Optional[Dict] = None

        if not execution_result.success:
            return [], None, None

        if not execution_result.stdout:
            return [], None, "Empty stdout — code produced no output"

        expected_steps = (
            {int(step_number) for step_number in expected_step_numbers}
            if expected_step_numbers is not None
            else None
        )
        seen_step_numbers: set = set()
        duplicate_steps: set = set()
        unexpected_steps: set = set()
        first_json_error: Optional[json.JSONDecodeError] = None

        for raw_line in execution_result.stdout.strip().split('\n'):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if first_json_error is None:
                    first_json_error = e
                continue

            if "final_answer_verified" in obj:
                if isinstance(obj.get("final_answer_verified"), bool):
                    final_verdict = obj
            elif "step" in obj:
                try:
                    step_num = int(obj["step"])
                except (TypeError, ValueError):
                    continue
                if not isinstance(obj.get("verified"), bool):
                    continue
                if step_num in seen_step_numbers:
                    duplicate_steps.add(step_num)
                    continue
                if expected_steps is not None and step_num not in expected_steps:
                    unexpected_steps.add(step_num)
                    continue
                seen_step_numbers.add(step_num)
                steps.append(StepVerification(
                    step_number=step_num,
                    description=obj.get("description", ""),
                    verified=obj["verified"],
                    note=obj.get("note", "")
                ))

        if not steps:
            if first_json_error is not None:
                return [], final_verdict, first_json_error
            return [], final_verdict, "No step verification lines found in output"

        if expected_steps is not None:
            missing_steps = expected_steps - seen_step_numbers
            alignment_errors = []
            if missing_steps:
                alignment_errors.append(
                    f"missing step output(s): {', '.join(str(n) for n in sorted(missing_steps))}"
                )
            if unexpected_steps:
                alignment_errors.append(
                    f"unexpected step output(s): {', '.join(str(n) for n in sorted(unexpected_steps))}"
                )
            if duplicate_steps:
                alignment_errors.append(
                    f"duplicate step output(s): {', '.join(str(n) for n in sorted(duplicate_steps))}"
                )
            if alignment_errors:
                return steps, final_verdict, "Step output contract violation: " + "; ".join(alignment_errors)

        steps.sort(key=lambda s: s.step_number)
        return steps, final_verdict, None
