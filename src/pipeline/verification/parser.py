from typing import List, Tuple, Optional, Dict, Any
import json
import re
from .verification_types import StepVerification, CodeExecutionResult


class VerificationOutputParser:
    def parse(
        self,
        execution_result: CodeExecutionResult
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

        seen_step_numbers: set = set()
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
                    continue  # Deduplicate
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

        steps.sort(key=lambda s: s.step_number)
        return steps, final_verdict, None
