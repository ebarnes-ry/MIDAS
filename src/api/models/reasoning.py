from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from .common import APIResponse


class ReasoningRequest(BaseModel):
    problem_statement: str = Field(..., description="The problem statement to reason about")
    visual_context: Optional[str] = Field(None, description="Visual context from vision analysis")
    source_metadata: Optional[Dict[str, Any]] = Field(None, description="Source metadata from vision pipeline")

    class Config:
        json_schema_extra = {
            "example": {
                "problem_statement": "Find the derivative of f(x) = x^2 + 3x + 1",
                "visual_context": "The function is a quadratic polynomial",
                "source_metadata": {}
            }
        }


class ReasoningStepResponse(BaseModel):
    step_number: int
    claim: str
    justification: str
    latex_expression: Optional[str] = None
    verification_status: Optional[bool] = None
    verification_note: Optional[str] = None
    feedback: Optional[str] = None


class FeedbackRequest(BaseModel):
    problem_statement: str
    step_number: int
    claim: str
    latex_expression: Optional[str] = None
    justification: str
    verification_note: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ReasoningData(BaseModel):
    original_problem: str
    steps: List[ReasoningStepResponse]
    worked_solution: str
    final_answer: str
    think_reasoning: str
    processing_time: float
    processing_metadata: Dict[str, Any] = {}


class ReasoningResponse(APIResponse):
    data: Optional[ReasoningData] = None


class ReasoningExplainRequest(BaseModel):
    problem_statement: str
    worked_solution: str
    step_text: str


class ReasoningExplainData(BaseModel):
    explanation: str
    processing_time: float


class ReasoningExplainResponse(APIResponse):
    data: Optional[ReasoningExplainData] = None


ReasoningResponse.model_rebuild()
ReasoningExplainResponse.model_rebuild()
