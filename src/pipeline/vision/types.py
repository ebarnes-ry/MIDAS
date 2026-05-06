from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Union, Literal

# Input types
@dataclass
class VisionInput:
    file_path: str
    file_type: str

@dataclass
class UserSelection:
    problem_id: str
    edited_latex: str
    original_image_path: str


class ProblemType(str, Enum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    PROOF = "proof"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    NUMBER_THEORY = "number_theory"
    LINEAR_ALGEBRA = "linear_algebra"
    OTHER = "other"


@dataclass
class Problem:
    problem_id: str
    problem_text: str
    figure_references: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)
    referenced_figure_descriptions: List[str] = field(default_factory=list)
    problem_type: ProblemType = ProblemType.OTHER

@dataclass
class UIBlock:
    id: str
    block_type: str
    html: str
    polygon: List[float]
    bbox: List[float]
    children: List['UIBlock']
    section_hierarchy: Dict[str, Any]
    images: Optional[Dict[str, str]] = None #Marker output images/visuals
    image_description: Optional[str] = None #Marker llm generated image/visual descriptions
    latex_content: Optional[str] = None
    is_editable: bool = False

@dataclass
class UIDocument:
    blocks: List[UIBlock]
    full_page_text: str
    images: Dict[str, str]
    metadata: Dict[str, Any]
    dimensions: Tuple[int, int]
    problems: List[Problem] = field(default_factory=list)

# VLM Schema types...
class Visual(BaseModel):
    description: str
    visual_type: Literal["Graph", "Plot", "Chart", "Figure", "Image", "Picture", "Table", "Matrix", "Grid", "Spreadsheet", "Diagram", "Geometry_Figure", "Geometric_Figure", "Shape", "Illustration", "Drawing", "Sketch"]
    data: Optional[Union[List[List[Any]], Dict[str, Any]]] = None

class VisualContext(BaseModel):
    elements: List[Visual] = []
    summary: Optional[str] = None
    contains_essential_info: bool

# Final Vision output
@dataclass
class VisionFinalOutput:
    problem_statement: str
    visual_context: Optional[VisualContext]
    source_metadata: Dict[str, Any]

class MathValidationResult(BaseModel):
    contains_math: bool = Field(..., description="True if the image contains a math problem, False otherwise.")
    reason: str = Field(..., description="A brief explanation for the decision.")