"""
API models for the vision pipeline endpoints.

These Pydantic models define the request/response schemas for document processing,
user interaction, and visual analysis. They serve as the contract between your
frontend and backend, separate from internal pipeline types.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple

from .common import APIResponse

# API Request Models
class UserSelectionRequest(BaseModel):
    document_id: str = Field(..., description="Document session ID")
    problem_id: str = Field(..., description="ID of the selected problem group")
    edited_latex: str = Field(..., description="User's edited LaTeX content")
    visual_context_override: Optional[str] = Field(None, description="User-edited visual context summary to attach to the selected problem")
    remove_visual_context: bool = Field(False, description="Whether to intentionally omit attached visual context")
    
    @validator('edited_latex')
    def validate_latex(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Edited LaTeX content cannot be empty")
        return v.strip()

# API Response Models  
class APIBlock(BaseModel):
    id: str
    block_type: str
    html: str
    polygon: List[float] # This MUST be a flat list of floats.
    bbox: List[float]
    is_editable: bool
    latex_content: Optional[str] = None
    cropped_image: Optional[str] = None
    image_description: Optional[str] = None

# class APIDocument(BaseModel):
#     """API representation of a processed document."""
#     blocks: List[APIBlock] = Field(..., description="All blocks in the document")
#     total_blocks: int = Field(..., description="Total number of blocks")
#     editable_blocks: int = Field(..., description="Number of editable blocks")  
#     images: Dict[str, str] = Field(default_factory=dict, description="Base64 encoded images")
#     dimensions: Tuple[int, int] = Field(..., description="Document dimensions (width, height)")
#     metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

# class APIProblem(BaseModel):
#     problem_id: str
#     block_ids: List[str]
#     combined_text: str

# class APIDocument(BaseModel):
#     """API representation of a processed document."""
#     blocks: List[APIBlock]
#     # === NEW: The list of problems is now part of the API response ===
#     problems: List[APIProblem]
#     total_blocks: int
#     editable_blocks: int
#     images: Dict[str, str]
#     dimensions: Tuple[int, int]
#     metadata: Dict[str, Any]

class APIProblem(BaseModel):
    problem_id: str
    problem_text: str
    block_ids: List[str]
    problem_type: Optional[str] = None
    figure_references: List[str] = Field(default_factory=list)
    visual_context_required: bool = False
    visual_context_attached: bool = False
    visual_context_summary: Optional[str] = None
    visual_context_description_count: int = 0

class APIDocument(BaseModel):
    blocks: List[APIBlock]
    problems: List[APIProblem]

class DocumentUploadResponse(APIResponse):
    """Response after successful document upload and processing."""
    data: Optional['DocumentUploadData'] = None

# class DocumentUploadData(BaseModel):
#     """Data payload for document upload response."""
#     document_id: str = Field(..., description="Unique document session ID")
#     document: APIDocument = Field(..., description="Processed document structure")
#     original_image_base64: str = Field(..., description="Original document as base64 image")  
#     processing_time: float = Field(..., description="Processing time in seconds")
#     processing_metadata: Dict[str, Any] = Field(..., description="Processing details")

class DocumentUploadData(BaseModel):
    """Data payload for document upload response."""
    document_id: str
    document: APIDocument # This now contains the problems list
    original_image_base64: str
    processing_time: float
    processing_metadata: Dict[str, Any]

class APIVisualElement(BaseModel):
    """API representation of a visual element."""
    description: str = Field(..., description="Description of the visual element")
    visual_type: str = Field(..., description="Type of visual (GRAPH, TABLE, etc.)")
    data: Optional[Any] = Field(None, description="Structured data if applicable")

class APIVisualContext(BaseModel):
    """API representation of visual context analysis."""
    elements: List[APIVisualElement] = Field(default_factory=list)
    summary: Optional[str] = Field(None, description="Summary of visual elements")
    contains_essential_info: bool = Field(..., description="Whether visuals are essential")

class VisionAnalysisResponse(APIResponse):
    """Response after visual analysis of selected content."""
    data: Optional['VisionAnalysisData'] = None

class VisionAnalysisData(BaseModel):
    """Data payload for vision analysis response."""
    problem_statement: str = Field(..., description="Final edited problem statement")
    visual_context: Optional[APIVisualContext] = Field(None, description="Visual analysis results")
    processing_time: float = Field(..., description="Analysis time in seconds")
    analysis_metadata: Dict[str, Any] = Field(..., description="Analysis details")

# Update forward references
DocumentUploadResponse.model_rebuild()
VisionAnalysisResponse.model_rebuild()
