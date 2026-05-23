"""
Trajectory logging for MIDAS.

Records the full repair trajectory for each problem: the original reasoning attempt,
each repair attempt with its verification result, and the final outcome.

Output: newline-delimited JSON (JSONL), one trajectory per line.
Schema per record:
{
  "trajectory_id": "<uuid>",
  "timestamp": "<iso8601>",
  "problem": { "statement": "...", "type": "...", "source": "..." },
  "attempts": [
    {
      "attempt_number": 1,
      "reasoning": { "steps": [...], "final_answer": "..." },
      "verification": { "status": "...", "steps_verified": N, "steps_failed": N, "errors": [...] },
      "generated_code": "..."
    }
  ],
  "outcome": {
    "final_status": "verified|failed",
    "attempt_count": 1,
    "difficulty_signal": 0.0   # 0.0=first-attempt success, 1.0=all attempts failed
  }
}
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AttemptRecord:
    attempt_number: int
    reasoning_steps: List[Dict]
    final_answer: str
    verification_status: str
    steps_verified: int
    steps_failed: int
    verification_errors: List[Dict]
    generated_code: str


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    timestamp: str
    problem_statement: str
    problem_type: str
    problem_source: str
    attempts: List[AttemptRecord] = field(default_factory=list)
    final_status: str = "pending"
    attempt_count: int = 0
    difficulty_signal: float = 0.0


class TrajectoryLogger:
    """
    Logs MIDAS proof trajectories to a JSONL file.

    Usage:
        logger = TrajectoryLogger("trajectories/midas_trajectories.jsonl")
        tid = logger.start_trajectory(problem_statement, problem_type)
        logger.log_attempt(tid, attempt_number, reasoning_output, verification_result)
        logger.close_trajectory(tid, final_status, max_attempts)
    """

    def __init__(self, log_path: str = "trajectories/midas_trajectories.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, TrajectoryRecord] = {}

    def start_trajectory(
        self,
        problem_statement: str,
        problem_type: str = "unknown",
        problem_source: str = "",
    ) -> str:
        tid = str(uuid.uuid4())
        self._active[tid] = TrajectoryRecord(
            trajectory_id=tid,
            timestamp=datetime.now(timezone.utc).isoformat(),
            problem_statement=problem_statement,
            problem_type=problem_type,
            problem_source=problem_source,
        )
        return tid

    def log_attempt(
        self,
        trajectory_id: str,
        attempt_number: int,
        reasoning_output: Any,
        verification_result: Any,
        generated_code: str = "",
    ) -> None:
        if trajectory_id not in self._active:
            return

        traj = self._active[trajectory_id]

        steps_data = [
            {
                "step_number": s.step_number,
                "claim": s.claim,
                "justification": s.justification,
                "latex_expression": s.latex_expression,
                "verification_status": s.verification_status,
                "verification_note": s.verification_note,
                "feedback": getattr(s, "feedback", None),
            }
            for s in getattr(reasoning_output, "steps", [])
        ]

        step_verifs = getattr(verification_result, "step_verifications", [])
        errors = []
        for e in getattr(verification_result, "errors", []):
            error_type = getattr(e, "error_type", "unknown")
            errors.append({
                "error_type": error_type.value if hasattr(error_type, "value") else str(error_type),
                "message": getattr(e, "message", str(e)),
            })

        traj.attempts.append(AttemptRecord(
            attempt_number=attempt_number,
            reasoning_steps=steps_data,
            final_answer=getattr(reasoning_output, "final_answer", ""),
            verification_status=verification_result.status,
            steps_verified=sum(1 for s in step_verifs if s.verified),
            steps_failed=sum(1 for s in step_verifs if not s.verified),
            verification_errors=errors,
            generated_code=generated_code,
        ))

    def close_trajectory(
        self,
        trajectory_id: str,
        final_status: str,
        max_attempts: int = 3,
    ) -> None:
        """Finalise and flush the trajectory to disk."""
        if trajectory_id not in self._active:
            return

        traj = self._active.pop(trajectory_id)
        traj.final_status = final_status
        traj.attempt_count = len(traj.attempts)

        # Difficulty signal: 0.0 = first-attempt success, 1.0 = all attempts failed
        if final_status == "verified" and traj.attempt_count > 0:
            traj.difficulty_signal = (traj.attempt_count - 1) / max(max_attempts, 1)
        else:
            traj.difficulty_signal = 1.0

        record = {
            "trajectory_id": traj.trajectory_id,
            "timestamp": traj.timestamp,
            "problem": {
                "statement": traj.problem_statement,
                "type": traj.problem_type,
                "source": traj.problem_source,
            },
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "reasoning": {
                        "steps": a.reasoning_steps,
                        "final_answer": a.final_answer,
                    },
                    "verification": {
                        "status": a.verification_status,
                        "steps_verified": a.steps_verified,
                        "steps_failed": a.steps_failed,
                        "errors": a.verification_errors,
                    },
                    "generated_code": a.generated_code,
                }
                for a in traj.attempts
            ],
            "outcome": {
                "final_status": traj.final_status,
                "attempt_count": traj.attempt_count,
                "difficulty_signal": traj.difficulty_signal,
            },
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_trajectories(self, n: int = 50) -> List[Dict]:
        """Return the N most recent trajectories, newest first."""
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(reversed(records))

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics over all logged trajectories."""
        records = self.read_trajectories(n=10000)
        if not records:
            return {"total": 0}

        verified = [r for r in records if r["outcome"]["final_status"] == "verified"]
        by_type: Dict[str, int] = {}
        difficulty_sum = 0.0

        for r in records:
            ptype = r["problem"]["type"]
            by_type[ptype] = by_type.get(ptype, 0) + 1
            difficulty_sum += r["outcome"]["difficulty_signal"]

        return {
            "total": len(records),
            "verified": len(verified),
            "failed": len(records) - len(verified),
            "success_rate": len(verified) / len(records),
            "mean_difficulty": difficulty_sum / len(records),
            "mean_attempts": sum(r["outcome"]["attempt_count"] for r in records) / len(records),
            "by_type": by_type,
        }
