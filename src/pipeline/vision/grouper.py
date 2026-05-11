# import json
# from typing import List, Dict, Tuple, Optional
# from pydantic import BaseModel, Field
# from dataclasses import dataclass

# from src.models.manager import ModelManager
# from .types import UIDocument, UIBlock, Problem

# # Pydantic model for robust parsing of the new, simpler LLM response
# class ProblemSchema(BaseModel):
#     problem_text: str
#     figure_references: List[str] = Field(default_factory=list)

# class GroupingResponse(BaseModel):
#     problems: List[ProblemSchema]

# @dataclass
# class GrouperResult:
#     """Result containing both the parsed problems and raw model output"""
#     problems: List[Problem]
#     raw_model_output: str
#     success: bool
#     error_message: Optional[str] = None

# class SemanticGrouper:
#     def __init__(self, model_manager: ModelManager):
#         self.model_manager = model_manager

#     @staticmethod
#     def _repair_tex_escapes(s: str) -> str:
#         """If JSON under-escaped TeX, \t and \r arrive as TAB/CR. Put them back visibly."""
#         if not s:
#             return s
#         return s.replace('\r', r'\r').replace('\t', r'\t')

#     def group(self, full_page_text: str) -> List[Problem]:
#         print("--- Starting semantic grouping of full page text ---")
#         if not full_page_text.strip():
#             return []

#         try:
#             response = self.model_manager.call(
#                 task="group_problems",
#                 prompt_ref="vision/group_problems@v2",
#                 variables={"full_page_text": full_page_text},
#                 schema=GroupingResponse
#             )

#             # if response.parsed and isinstance(response.parsed, GroupingResponse):
#             #     print(f"Semantic grouping successful. Found {len(response.parsed.problems)} problems.")
#             #     return [
#             #         Problem(
#             #             problem_id=f"problem_{i+1}",
#             #             problem_text=p.problem_text,
#             #             figure_references=p.figure_references
#             #         ) for i, p in enumerate(response.parsed.problems)
#             #     ]
#             if response.parsed and isinstance(response.parsed, GroupingResponse):
#                 problems: List[Problem] = []
#                 for i, p in enumerate(response.parsed.problems):
#                     fixed = self._repair_tex_escapes(p.problem_text)
#                     problems.append(Problem(
#                         problem_id=f"problem_{i+1}",
#                         problem_text=fixed,
#                         figure_references=p.figure_references
#                     ))
#                 return problems
#             else:
#                 print(f" Semantic grouping failed to parse. Raw response: {response.content}")
#                 return []
#         except Exception as e:
#             print(f"An exception occurred during semantic grouping: {e}")
#             return []

import json
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

from src.models.manager import ModelManager
from .types import UIDocument, UIBlock, Problem, ProblemType

# Pydantic model for robust parsing of the new, simpler LLM response
class ProblemSchema(BaseModel):
    problem_text: str
    figure_references: List[str] = Field(default_factory=list)

class GroupingResponse(BaseModel):
    problems: List[ProblemSchema]

@dataclass
class GrouperResult:
    """Result containing both the parsed problems and raw model output"""
    problems: List[Problem]
    raw_model_output: str
    success: bool
    error_message: Optional[str] = None

class SemanticGrouper:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def _task_prompt_ref(self, task: str, default: str) -> str:
        config = getattr(self.model_manager, "config", {}) or {}
        return (
            config.get("tasks", {})
            .get(task, {})
            .get("prompt_ref")
            or default
        )

    def _classify_problem_type(self, problem_text: str) -> ProblemType:
        """Heuristic keyword-based classification of a math problem."""
        text = problem_text.lower()
        if any(k in text for k in ["prove", "proof", "show that", "demonstrate", "if and only if", "∀", "∃"]):
            return ProblemType.PROOF
        if any(k in text for k in ["∫", "integral", "derivative", "differentiate", "lim", "limit", "∂", "series", "converge"]):
            return ProblemType.CALCULUS
        if any(k in text for k in ["matrix", "vector", "eigenvalue", "determinant", "span", "basis"]):
            return ProblemType.LINEAR_ALGEBRA
        if any(k in text for k in ["probability", "expected value", "variance", "distribution", "p(x"]):
            return ProblemType.STATISTICS
        if any(k in text for k in ["prime", "divisible", "modulo", "congruent", "integer"]):
            return ProblemType.NUMBER_THEORY
        if any(k in text for k in ["triangle", "angle", "circle", "area", "perimeter", "geometric"]):
            return ProblemType.GEOMETRY
        if any(k in text for k in ["solve", "equation", "simplify", "factor", "expand", "polynomial"]):
            return ProblemType.ALGEBRA
        return ProblemType.OTHER

    def group(self, full_page_text: str) -> List[Problem]:
        print("--- Starting semantic grouping of full page text ---")
        if not full_page_text.strip():
            return []

        try:
            prompt_ref = self._task_prompt_ref("group_problems", "vision/group_problems@v2")
            response = self.model_manager.call(
                task="group_problems",
                prompt_ref=prompt_ref,
                variables={"full_page_text": full_page_text},
                schema=GroupingResponse
            )

            if response.parsed and isinstance(response.parsed, GroupingResponse):
                print(f"Semantic grouping successful. Found {len(response.parsed.problems)} problems.")
                problems = []
                for i, p in enumerate(response.parsed.problems):
                    prob = Problem(
                        problem_id=f"problem_{i+1}",
                        problem_text=p.problem_text,
                        figure_references=p.figure_references,
                    )
                    prob.problem_type = self._classify_problem_type(p.problem_text)
                    problems.append(prob)
                return problems
            else:
                print(f" Semantic grouping failed to parse. Raw response: {response.content}")
                return []
        except Exception as e:
            print(f"An exception occurred during semantic grouping: {e}")
            return []


# # Pydantic model for robust parsing of the LLM response
# class ProblemSchema(BaseModel):
#     problem_id: str
#     block_ids: List[str]
#     combined_text: str
#     figure_references: List[str] = Field(default_factory=list)

# class GroupingResponse(BaseModel):
#     problems: List[ProblemSchema]

# class SemanticGrouper:
#     """
#     Uses an LLM to group raw OCR blocks into semantically complete problems.
#     This is the core of the new, sane architecture.
#     """
#     def __init__(self, model_manager: ModelManager):
#         self.model_manager = model_manager

#     def group(self, document: UIDocument) -> List[Problem]:
#         """
#         Takes a document with raw blocks and returns a list of identified problems.
#         """
#         print("--- Starting semantic grouping of document blocks ---")
#         if not document.blocks:
#             return []

#         # In grouper.py (Corrected)
#         block_contexts = [
#             {"id": b.id, "latex_content": b.latex_content, "html": b.html}
#             for b in document.blocks if b.latex_content or b.html and '<content-ref' not in b.html
#         ]

#         print("--- INPUT TO SEMANTIC GROUPER LLM ---")
#         print(json.dumps(block_contexts, indent=2))
#         print("------------------------------------")

#         try:
#             response = self.model_manager.call(
#                 task="group_problems",
#                 prompt_ref="vision/group_problems@v1",
#                 variables={"blocks": block_contexts},
#                 schema=GroupingResponse
#             )

#             print("--- RAW LLM OUTPUT ---")
#             print(response.content)
#             print("----------------------")

#             if response.parsed and isinstance(response.parsed, GroupingResponse):
#                 print(f"Semantic grouping successful. Found {len(response.parsed.problems)} problems.")
#                 # Convert from Pydantic models to dataclasses
#                 return [
#                     Problem(
#                         problem_id=p.problem_id,
#                         block_ids=p.block_ids,
#                         combined_text=p.combined_text,
#                         figure_references=p.figure_references
#                     ) for p in response.parsed.problems
#                 ]
#             else:
#                 print(f"Semantic grouping failed to parse. Raw response: {response.content}")
#                 # Even if parsing fails, don't crash. Return no problems.
#                 return []
#         except Exception as e:
#             print(f"An exception occurred during semantic grouping: {e}")
#             return []
