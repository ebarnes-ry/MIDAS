from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

if TYPE_CHECKING:
    from ..reasoning.types import ReasoningOutput

class ErrorType(Enum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    ASSERTION_FAILED = "assertion_failed"
    ANSWER_MISMATCH = "answer_mismatch"
    INCOMPLETE_VERIFICATION = "incomplete_verification"
    SYMBOLIC_FAILURE = "symbolic_failure"
    CONTRACT_VIOLATION = "contract_violation"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED_REASONING = "failed_reasoning"
    FAILED_CODEGEN = "failed_codegen"
    FAILED_CONTRACT = "failed_contract"
    UNSUPPORTED = "unsupported"
    FAILED_PIPELINE = "failed_pipeline"

    def __str__(self) -> str:
        return self.value

class VerificationError(BaseModel):
    error_type: ErrorType
    message: str
    line_number: Optional[int] = None
    problematic_code: Optional[str] = None
    suggested_fix: Optional[str] = None
    traceback: Optional[str] = None

class StepVerification(BaseModel):
    step_number: int
    description: str
    verified: bool
    note: str = ""

class CodeExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    execution_time: float
    namespace: Dict[str, Any] = Field(default_factory=dict)
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None

class VerificationResult(BaseModel):
    """The complete, final outcome of the verification pipeline."""
    status: VerificationStatus
    confidence_score: float
    reasoning_output: Any  # Can hold the ReasoningOutput object
    generated_code: str
    execution_result: Optional[CodeExecutionResult] = None
    step_verifications: List[StepVerification] = Field(default_factory=list)
    answer_match: Optional[bool] = None
    errors: List[VerificationError] = Field(default_factory=list)
    repair_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
