"""
Unit tests for step-level verification alignment (STEP_02).
Verifies that _annotate_reasoning_steps writes verification results
back onto ReasoningStep objects correctly.
"""
import pytest
from src.pipeline.reasoning.types import ReasoningOutput, ReasoningStep
from src.pipeline.verification.verification_types import StepVerification, CodeExecutionResult
from src.pipeline.verification.parser import VerificationOutputParser
from src.pipeline.verification.verification import VerificationPipeline


def _make_reasoning_output(num_steps: int = 3) -> ReasoningOutput:
    steps = [
        ReasoningStep(
            step_number=i + 1,
            claim=f"claim for step {i + 1}",
            justification=f"justification {i + 1}",
            latex_expression=f"$x = {i + 1}$",
        )
        for i in range(num_steps)
    ]
    return ReasoningOutput(
        original_problem="Solve 3x = 6",
        steps=steps,
        final_answer="2",
        think_reasoning="",
    )


def _make_step_verifications(step_results: list) -> list:
    """step_results: list of (step_number, verified, note)"""
    return [
        StepVerification(step_number=n, description=f"claim for step {n}", verified=v, note=note)
        for n, v, note in step_results
    ]


class TestAnnotateReasoningSteps:
    def _pipeline(self):
        """VerificationPipeline without a real ModelManager — we only test _annotate_reasoning_steps."""
        p = object.__new__(VerificationPipeline)
        return p

    def test_all_steps_annotated(self):
        reasoning = _make_reasoning_output(3)
        svs = _make_step_verifications([(1, True, ""), (2, True, ""), (3, False, "SymPy gives x=4, not x=3")])

        self._pipeline()._annotate_reasoning_steps(reasoning, svs)

        assert reasoning.steps[0].verification_status is True
        assert reasoning.steps[0].verification_note is None
        assert reasoning.steps[1].verification_status is True
        assert reasoning.steps[1].verification_note is None
        assert reasoning.steps[2].verification_status is False
        assert reasoning.steps[2].verification_note == "SymPy gives x=4, not x=3"

    def test_partial_coverage_leaves_unmatched_as_none(self):
        reasoning = _make_reasoning_output(3)
        svs = _make_step_verifications([(1, True, ""), (3, False, "mismatch")])

        self._pipeline()._annotate_reasoning_steps(reasoning, svs)

        assert reasoning.steps[0].verification_status is True
        assert reasoning.steps[1].verification_status is None  # not in svs — stays None
        assert reasoning.steps[2].verification_status is False

    def test_empty_verifications_leaves_all_none(self):
        reasoning = _make_reasoning_output(3)
        self._pipeline()._annotate_reasoning_steps(reasoning, [])
        assert all(s.verification_status is None for s in reasoning.steps)

    def test_verified_note_is_cleared(self):
        reasoning = _make_reasoning_output(1)
        reasoning.steps[0].verification_note = "stale note"
        svs = _make_step_verifications([(1, True, "confirmed by SymPy")])

        self._pipeline()._annotate_reasoning_steps(reasoning, svs)

        # Verified steps should have note cleared to None
        assert reasoning.steps[0].verification_note is None


class TestVerificationOutputParser:
    def _exec_result(self, stdout: str) -> CodeExecutionResult:
        return CodeExecutionResult(success=True, stdout=stdout, stderr="", execution_time=0.1)

    def test_parses_step_and_final_verdict(self):
        stdout = (
            '{"step": 1, "description": "x equals 2", "verified": true, "note": "confirmed"}\n'
            '{"step": 2, "description": "2x equals 4", "verified": false, "note": "SymPy gives 3"}\n'
            '{"final_answer_verified": false, "answer": "2", "note": "mismatch"}\n'
        )
        parser = VerificationOutputParser()
        steps, verdict, err = parser.parse(self._exec_result(stdout))

        assert err is None
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert steps[0].verified is True
        assert steps[0].note == "confirmed"
        assert steps[1].step_number == 2
        assert steps[1].verified is False
        assert steps[1].note == "SymPy gives 3"
        assert verdict["final_answer_verified"] is False

    def test_skips_non_json_lines(self):
        stdout = (
            "some debug output\n"
            '{"step": 1, "description": "step 1", "verified": true, "note": ""}\n'
            "another random line\n"
            '{"final_answer_verified": true, "answer": "3", "note": ""}\n'
        )
        parser = VerificationOutputParser()
        steps, verdict, err = parser.parse(self._exec_result(stdout))

        assert err is None
        assert len(steps) == 1
        assert verdict is not None

    def test_deduplicates_repeated_step_numbers(self):
        stdout = (
            '{"step": 1, "description": "first", "verified": true, "note": ""}\n'
            '{"step": 1, "description": "duplicate", "verified": false, "note": ""}\n'
            '{"final_answer_verified": true, "answer": "x", "note": ""}\n'
        )
        parser = VerificationOutputParser()
        steps, _, _ = parser.parse(self._exec_result(stdout))

        assert len(steps) == 1
        assert steps[0].verified is True  # First occurrence wins

    def test_empty_stdout_returns_error(self):
        parser = VerificationOutputParser()
        steps, verdict, err = parser.parse(self._exec_result(""))

        assert steps == []
        assert verdict is None
        assert err is not None

    def test_steps_sorted_by_number(self):
        stdout = (
            '{"step": 3, "description": "c", "verified": true, "note": ""}\n'
            '{"step": 1, "description": "a", "verified": true, "note": ""}\n'
            '{"step": 2, "description": "b", "verified": true, "note": ""}\n'
            '{"final_answer_verified": true, "answer": "x", "note": ""}\n'
        )
        parser = VerificationOutputParser()
        steps, _, _ = parser.parse(self._exec_result(stdout))

        assert [s.step_number for s in steps] == [1, 2, 3]
