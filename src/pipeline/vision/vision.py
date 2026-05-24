from src.models.manager import ModelManager
from .types import Problem, VisionInput, UserSelection, VisionFinalOutput, UIDocument, VisualContext
from .ui_transformer import UITransformer
from .vlm import VisualContextualizer
from .grouper import SemanticGrouper

from typing import Dict, Union, Optional, List, Tuple
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher
from pydantic import BaseModel, ConfigDict, Field
import re

class ImageProblemRecovery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovered_problem_text: str = Field(
        ...,
        description="The complete mathematical problem text transcribed from the image, including required equations, matrices, integrals, or expressions.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)

class VisionPipeline:
    def __init__(self, manager: ModelManager):
        self.model_manager = manager
        self._marker_service = None
        #self.visual_contextualizer = VisualContextualizer(manager)
        self.grouper = SemanticGrouper(manager)

    @property
    def marker_service(self):
        if self._marker_service is None:
            self._marker_service = self.model_manager.marker
        return self._marker_service

    #def process_input(self, vision_input: VisionInput) -> UIDocument:
    def process_input(self, vision_input: VisionInput) -> UIDocument:
        """The main entry point for processing an uploaded document."""
        # Step 1: OCR with Marker (no LLM — fast path).
        marker_result = self.marker_service.convert_document(vision_input.file_path)
        if marker_result is None:
            raise ValueError("Marker processing failed")

        # Step 2: Transform Marker output to UIDocument.
        ui_document = UITransformer.transform_marker_json(marker_result)

        # Step 3: Recover math from any blocks Marker misclassified as Picture/Figure.
        # Marker's LLM mode would do this per-block with Gemini (slow); we do it in one
        # Groq call across all affected blocks.
        recovered = self._recover_image_block_text(ui_document)
        if recovered:
            # Append recovered math text to full_page_text so the grouper can see it.
            ui_document.full_page_text = (ui_document.full_page_text + "\n\n" + recovered).strip()

        problems_with_blocks = self._assemble_problems(ui_document)
        problems_with_blocks = self._recover_incomplete_problem_text_from_image(
            problems_with_blocks,
            ui_document,
            vision_input.file_path,
        )

        # Step 5: Explicitly associate figure descriptions with the problems.
        ui_document.problems = self._associate_descriptions_to_problems(problems_with_blocks, ui_document)
        
        return ui_document

    def _assemble_problems(self, ui_document: UIDocument) -> List[Problem]:
        problems = self.grouper.group_document(ui_document)
        problems_with_blocks = self._link_problems_to_blocks(problems, ui_document)
        problems_with_blocks = self._repair_problem_assembly(problems_with_blocks, ui_document)
        problems_with_blocks = self._repair_problem_text_from_linked_blocks(problems_with_blocks, ui_document)
        problems_with_blocks = self._reclassify_problem_types(problems_with_blocks, ui_document)
        return self._annotate_problem_completeness(problems_with_blocks)

    def _recover_image_block_text(self, doc: UIDocument) -> str:
        """
        Replaces Marker's per-block Gemini calls with a single Groq call.

        When use_llm=False, Marker leaves Picture/Figure blocks with no text content.
        These are often misclassified math equations. We collect all such blocks,
        describe them in one prompt, and ask the model to extract any maths.

        Returns a string to append to full_page_text, or "" if nothing to recover.
        """
        orphan_descriptions = [
            block.image_description
            for block in doc.blocks
            if block.block_type.lower() in {"figure", "picture"}
            and not block.latex_content
            and block.image_description
        ]

        # No image descriptions means use_llm was off AND no fallback descriptions exist.
        # We can't recover without image data at this stage — return empty.
        if not orphan_descriptions:
            return ""

        combined = "\n\n".join(f"Block {i+1}: {d}" for i, d in enumerate(orphan_descriptions))
        prompt = (
            "The following are descriptions of image blocks extracted from a maths document. "
            "For each block, extract any mathematical content and rewrite it with proper LaTeX "
            "delimiters ($...$ for inline, $$...$$ for display). "
            "Return only the extracted maths, one block per line. "
            "If a block contains no maths, output 'NONE'.\n\n" + combined
        )

        try:
            response = self.model_manager.call(
                task="group_problems",   # reuses the same fast Groq text model
                prompt_ref="vision/group_problems@v3",
                variables={},
                messages_override=[
                    {"role": "system", "content": "You are a maths OCR corrector. Extract and LaTeX-format any mathematical content from image descriptions."},
                    {"role": "user",   "content": prompt},
                ]
            )
            lines = [l.strip() for l in response.content.strip().split("\n") if l.strip() and l.strip() != "NONE"]
            return "\n".join(lines)
        except Exception as e:
            print(f"[vision] image block recovery skipped: {e}")
            return ""

    def _normalize_text(self, text: str) -> str:
        """A helper to clean text for robust comparison."""
        if not text:
            return ""
        # Lowercase, remove all non-alphanumeric characters, and collapse whitespace
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _link_problems_to_blocks(self, problems: List[Problem], document: UIDocument) -> List[Problem]:
        """
        Associate text-based problems with the UI blocks they originated from
        using a robust similarity matching algorithm.
        """
        print("\n--- Starting Block to Problem Linking Process ---")
        normalized_problems = { p.problem_id: self._normalize_text(p.problem_text) for p in problems }
        block_assignments: Dict[str, Tuple[str, float]] = {}

        for block in document.blocks:
            block_text = block.latex_content
            # print("\n\n\n\n\n\n\n\n")
            # print(block_text)
            if not block_text or not block_text.strip():
                continue

            normalized_block_text = self._normalize_text(block_text)
            if not normalized_block_text:
                continue

            # print("\n\n\n\n\n\n\n\n")
            # print(normalized_block_text)
            print(f"\n[DEBUG] Analyzing Block ID: {block.id}")
            print(f"  - Normalized Block Text: '{normalized_block_text}'")
            

            best_ratio = 0.0
            best_problem_id = None

            for problem_id, normalized_problem_text in normalized_problems.items():
                print(f"  - Normalized Problem Text: '{normalized_problem_text}'")
                if normalized_block_text in normalized_problem_text:
                    ratio = 1.0 # Perfect substring match
                else:
                    matcher = SequenceMatcher(None, normalized_block_text, normalized_problem_text)
                    match = matcher.find_longest_match(0, len(normalized_block_text), 0, len(normalized_problem_text))
                    ratio = match.size / len(normalized_block_text)

                print(f"  - Comparing with {problem_id}: Ratio = {ratio:.2f}")

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_problem_id = problem_id
            
            # Use a slightly more lenient threshold and log the decision
            #threshold = 0.85
            threshold = 0.70
            if best_problem_id and best_ratio > threshold:
                block_assignments[block.id] = (best_problem_id, best_ratio)
                print(f"  ASSIGNED to {best_problem_id} (Ratio: {best_ratio:.2f} > {threshold})")
            else:
                print(f"  NOT ASSIGNED (Best Ratio: {best_ratio:.2f} <= {threshold})")

        valid_block_ids = {block.id for block in document.blocks}

        # Second pass: Assign blocks to problems. Keep explicit block ids returned
        # by block-aware grouping and use fuzzy matching to fill any gaps.
        for problem in problems:
            existing_ids = [block_id for block_id in problem.block_ids if block_id in valid_block_ids]
            matched_ids = [
                block_id for block_id, (p_id, ratio) in block_assignments.items()
                if p_id == problem.problem_id
            ]
            problem.block_ids = list(dict.fromkeys(existing_ids + matched_ids))
        
        print("--- Block to Problem Linking Complete ---")
        return problems

    def _block_order(self, document: UIDocument) -> Dict[str, int]:
        return {block.id: index for index, block in enumerate(document.blocks)}

    def _block_text(self, block) -> str:
        return (block.latex_content or "").strip()

    def _linked_block_latex(self, problem: Problem, document: UIDocument) -> List[str]:
        linked_ids = set(problem.block_ids)
        return [
            block.latex_content
            for block in document.blocks
            if block.id in linked_ids and block.latex_content
        ]

    def _linked_block_text_in_order(self, problem: Problem, document: UIDocument) -> str:
        linked_ids = set(problem.block_ids)
        return "\n".join(
            block.latex_content.strip()
            for block in document.blocks
            if block.id in linked_ids
            and block.latex_content
            and block.latex_content.strip()
        ).strip()

    def _has_important_math_missing_from_problem(self, candidate: str, problem_text: str) -> bool:
        candidate_lower = candidate.lower()
        problem_lower = problem_text.lower()
        math_markers = (
            "\\begin",
            "\\end",
            "\\frac",
            "\\sqrt",
            "\\sum",
            "\\int",
            "\\lim",
            "\\det",
            "bmatrix",
            "pmatrix",
            "matrix",
            "=",
            "^",
            "_",
        )
        return any(marker in candidate_lower and marker not in problem_lower for marker in math_markers)

    def _repair_problem_text_from_linked_blocks(self, problems: List[Problem], document: UIDocument) -> List[Problem]:
        """
        Prefer linked block text when grouping returned an incomplete stem but
        block linking found a richer block containing the missing math. This is
        intentionally conservative: the linked text must contain the grouped
        text and add math markers absent from the grouped problem statement.
        """
        for problem in problems:
            candidate = self._linked_block_text_in_order(problem, document)
            if not candidate:
                continue

            normalized_problem = self._normalize_text(problem.problem_text)
            normalized_candidate = self._normalize_text(candidate)
            if not normalized_problem or normalized_problem == normalized_candidate:
                continue

            contains_problem = normalized_problem in normalized_candidate
            materially_richer = len(normalized_candidate) > len(normalized_problem) + 8
            missing_math = self._has_important_math_missing_from_problem(candidate, problem.problem_text)

            if contains_problem and materially_richer and missing_math:
                problem.problem_text = candidate
        return problems

    def _reclassify_problem_types(self, problems: List[Problem], document: UIDocument) -> List[Problem]:
        """
        Re-run problem type classification after block linking/repair so raw
        LaTeX that was not present in the grouped text can influence metadata.
        """
        for problem in problems:
            problem.problem_type = self.grouper._classify_problem_type(
                problem.problem_text,
                self._linked_block_latex(problem, document),
            )
        return problems

    def _problem_missing_content_reason(self, problem: Problem) -> Optional[str]:
        text = (problem.problem_text or "").strip()
        normalized = self._normalize_text(text)
        if not normalized:
            return "empty_problem_text"

        has_matrix_context = any(
            marker in normalized
            for marker in ("matrix", "matrices", "eigenvalue", "eigenvalues", "determinant")
        )
        has_matrix_content = bool(
            re.search(r"\\begin\{[bpv]?matrix\}", text)
            or re.search(r"\\begin\{array\}", text)
            or re.search(r"\[\s*[-\d\\&\s]+\s*\\\\\s*[-\d\\&\s]+\]", text)
        )
        if has_matrix_context and not has_matrix_content:
            return "instruction_without_matrix"

        has_system_instruction = "solve the system" in normalized or normalized.startswith("system")
        has_system_content = (
            "\\begin{cases}" in text
            or "\\begin{aligned}" in text
            or "\\begin{align}" in text
            or text.count("=") >= 2
        )
        if has_system_instruction and not has_system_content:
            return "instruction_without_equations"

        has_evaluate_instruction = normalized.startswith("evaluate")
        has_calculus_content = any(marker in text for marker in ("\\int", "∫", "\\lim", "\\frac{d", "="))
        if has_evaluate_instruction and not has_calculus_content:
            return "instruction_without_expression"

        return None

    def _looks_like_bare_single_equation(self, problem: Problem) -> bool:
        text = (problem.problem_text or "").strip()
        normalized = self._normalize_text(text)
        if not text or not normalized:
            return False
        instruction_words = (
            "solve",
            "find",
            "evaluate",
            "determine",
            "calculate",
            "simplify",
            "factor",
            "differentiate",
            "integrate",
            "prove",
            "show",
        )
        if any(word in normalized.split() for word in instruction_words):
            return False
        return text.count("=") == 1 and len(normalized.split()) <= 6

    def _annotate_problem_completeness(self, problems: List[Problem]) -> List[Problem]:
        for problem in problems:
            reason = self._problem_missing_content_reason(problem)
            problem.problem_input_complete = reason is None
            problem.missing_problem_content = reason is not None
            problem.missing_content_reason = reason
        return problems

    def _needs_image_text_recovery(self, problems: List[Problem]) -> bool:
        if any(self._problem_missing_content_reason(problem) for problem in problems):
            return True
        return len(problems) == 1 and self._looks_like_bare_single_equation(problems[0])

    def _recovered_text_is_better(self, current_text: str, recovered_text: str) -> bool:
        normalized_current = self._normalize_text(current_text)
        normalized_recovered = self._normalize_text(recovered_text)
        if not normalized_recovered or normalized_recovered == normalized_current:
            return False
        if len(normalized_recovered) <= len(normalized_current) + 4:
            return False

        has_current = normalized_current and normalized_current in normalized_recovered
        adds_structural_math = self._has_important_math_missing_from_problem(recovered_text, current_text)
        adds_instruction = any(
            word in normalized_recovered and word not in normalized_current
            for word in ("solve", "system", "find", "evaluate", "matrix", "eigenvalues")
        )
        return has_current or adds_structural_math or adds_instruction

    def _recover_problem_text_from_source_image(self, image_path: str, current_text: str) -> Optional[str]:
        prompt = (
            "Transcribe the complete mathematical problem from the image. "
            "Include the instruction words and all required equations, matrices, integrals, expressions, choices, or givens. "
            "Use LaTeX for mathematical notation, including matrix and cases environments when appropriate. "
            "Do not include worked solution text or answer text if it is clearly separate from the question. "
            "If the current extracted text below is incomplete, correct it using the image. "
            "Return only the complete problem statement in recovered_problem_text.\n\n"
            f"CURRENT EXTRACTED TEXT:\n{current_text or '[empty]'}"
        )
        try:
            with Image.open(image_path) as image:
                response = self.model_manager.call(
                    task="validation",
                    prompt_ref="vision/validate@v1",
                    variables={},
                    messages_override=[
                        {
                            "role": "system",
                            "content": "You are a precise math OCR recovery agent. Transcribe only the problem statement visible in the image.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    schema=ImageProblemRecovery,
                    images=[image.convert("RGB")],
                    temperature=0,
                    max_tokens=700,
                )
        except Exception as e:
            print(f"[vision] source-image text recovery skipped: {e}")
            return None

        if not response.parsed:
            return None
        recovered = response.parsed.recovered_problem_text.strip()
        if response.parsed.confidence < 0.55:
            return None
        return recovered or None

    def _recover_incomplete_problem_text_from_image(
        self,
        problems: List[Problem],
        document: UIDocument,
        image_path: str,
    ) -> List[Problem]:
        if not problems or not self._needs_image_text_recovery(problems):
            return self._annotate_problem_completeness(problems)

        # Whole-image recovery is intentionally limited to the single-problem
        # case. For multi-problem pages, replacing one grouped problem with a
        # full-page transcription could mix unrelated problems.
        if len(problems) != 1:
            return self._annotate_problem_completeness(problems)

        problem = problems[0]
        recovered_text = self._recover_problem_text_from_source_image(
            image_path,
            problem.problem_text,
        )
        if recovered_text and self._recovered_text_is_better(problem.problem_text, recovered_text):
            problem.problem_text = recovered_text
            problem.extraction_recovery_source = "source_image_ocr"
            problem.problem_type = self.grouper._classify_problem_type(
                problem.problem_text,
                self._linked_block_latex(problem, document),
            )

        return self._annotate_problem_completeness(problems)

    def _is_math_like_block(self, block) -> bool:
        block_type = block.block_type.lower()
        text = self._block_text(block)
        if not text:
            return False
        if block_type in {"equation", "inlinemath", "handwriting"}:
            return True
        math_markers = ("\\", "=", "^", "_", "+", "-", "\\int", "\\begin", "matrix")
        return any(marker in text for marker in math_markers)

    def _is_instruction_stem(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        stem_markers = (
            "solve the system",
            "evaluate",
            "find the derivative of",
            "find the eigenvalues of",
            "find the eigenvalue of",
            "differentiate",
            "integrate",
        )
        if not any(normalized.startswith(marker) for marker in stem_markers):
            return False
        return not any(marker in text for marker in ("\\", "=", "^", "_", "+", "-", "∫"))

    def _repair_problem_assembly(self, problems: List[Problem], document: UIDocument) -> List[Problem]:
        """
        Conservatively repair common Marker splits where an instruction stem is
        immediately followed by one or more math blocks. This only merges adjacent
        math-like blocks and never crosses another text instruction.
        """
        if not problems or not document.blocks:
            return problems

        order = self._block_order(document)

        for problem in problems:
            if not self._is_instruction_stem(problem.problem_text):
                continue

            linked_indices = [
                order[block_id]
                for block_id in problem.block_ids
                if block_id in order
            ]
            if not linked_indices:
                continue

            next_index = max(linked_indices) + 1
            added_ids: List[str] = []
            added_texts: List[str] = []

            while next_index < len(document.blocks):
                block = document.blocks[next_index]
                if not self._is_math_like_block(block):
                    break
                added_ids.append(block.id)
                added_texts.append(self._block_text(block))
                next_index += 1

            if not added_ids:
                continue

            problem.block_ids = list(dict.fromkeys(problem.block_ids + added_ids))
            suffix = "\n".join(text for text in added_texts if text)
            if suffix and suffix not in problem.problem_text:
                problem.problem_text = f"{problem.problem_text.rstrip()}\n{suffix}"

        return problems

    def _problem_appears_visual_dependent(self, problem: Problem) -> bool:
        text = problem.problem_text.lower()
        visual_markers = (
            "figure",
            "diagram",
            "graph",
            "table",
            "chart",
            "shown",
            "below",
            "above",
            "image",
            "picture",
        )
        return problem.problem_type == "geometry" or any(marker in text for marker in visual_markers)

    def _looks_like_text_or_math_fragment(self, description: str) -> bool:
        if not description:
            return False

        text = description.lower()
        fragment_markers = (
            "close-up",
            "close up",
            "segment",
            "part of",
            "portion",
            "fragment",
            "single large",
            "single digit",
            "digit",
            "letter",
            "text",
            "notation",
            "expression",
            "serif font",
            "italicized",
            "background",
            "differential",
        )
        math_text_markers = (
            "dx",
            "dy",
            "equation",
            "integral",
            "derivative",
            "mathematical expression",
        )
        return any(marker in text for marker in fragment_markers) and any(
            marker in text for marker in math_text_markers
        )

    def _looks_like_meaningful_visual_description(self, description: str) -> bool:
        if not description:
            return False

        text = description.lower()
        positive_visual_markers = (
            "graph",
            "plot",
            "axis",
            "axes",
            "coordinate plane",
            "table",
            "chart",
            "diagram",
            "figure",
            "triangle",
            "circle",
            "angle",
            "rectangle",
            "polygon",
            "line segment",
            "ray",
            "parallel",
            "perpendicular",
            "curve",
            "bar chart",
            "histogram",
            "scatter",
            "number line",
            "grid",
            "matrix",
        )
        return any(
            re.search(rf"\b{re.escape(marker)}\b", text)
            for marker in positive_visual_markers
        )

    def _drop_unmentioned_figure_references(self, problem: Problem) -> None:
        """
        The grouping model may infer references like "Picture 1" from Marker
        fragment descriptions that were appended to full_page_text. Keep only
        references that are actually present in the problem statement.
        """
        if not problem.figure_references:
            return

        normalized_problem = self._normalize_text(problem.problem_text)
        problem.figure_references = [
            reference
            for reference in problem.figure_references
            if self._normalize_text(reference)
            and self._normalize_text(reference) in normalized_problem
        ]

    def _description_overlaps_problem_text(self, description: str, problem: Problem, document: UIDocument) -> bool:
        """
        Detect descriptions that are just OCR/VLM narration of text already present
        in the problem text or nearby math/text blocks. These should remain
        visible as raw block metadata, but they should not become reasoning
        visual context.
        """
        normalized_description = self._normalize_text(description)
        if not normalized_description:
            return False

        candidate_texts = [problem.problem_text]
        linked_ids = set(problem.block_ids)
        candidate_texts.extend(
            block.latex_content or ""
            for block in document.blocks
            if block.id in linked_ids and block.latex_content
        )
        candidate_texts.extend(
            block.latex_content or ""
            for block in document.blocks
            if block.id not in linked_ids
            and block.latex_content
            and block.block_type.lower() in {"equation", "inlinemath"}
        )

        normalized_candidates = [
            self._normalize_text(text)
            for text in candidate_texts
            if self._normalize_text(text)
        ]

        for candidate in normalized_candidates:
            if len(candidate) >= 3 and candidate in normalized_description:
                return True
            if len(normalized_description) >= 3 and normalized_description in candidate:
                return True
            matcher = SequenceMatcher(None, normalized_description, candidate)
            if matcher.ratio() >= 0.72:
                return True

        return False

    def _is_meaningful_visual_context(self, description: str, problem: Problem, document: UIDocument) -> bool:
        """
        Keep only visual descriptions that carry non-textual problem information.
        Marker often emits Picture blocks for equation fragments; attaching those
        as visual context pollutes reasoning and makes symbolic problems appear
        diagram-dependent.
        """
        if not description or not description.strip():
            return False

        if self._looks_like_text_or_math_fragment(description):
            return False

        if self._description_overlaps_problem_text(description, problem, document):
            return False

        return self._looks_like_meaningful_visual_description(description)

    def _associate_descriptions_to_problems(self, problems: List[Problem], document: UIDocument) -> List[Problem]:
        """
        Linus's Note: I have rewritten this function to fix the silent failure.
        The old logic was too fragile. This version uses a robust heuristic.
        """
        described_blocks = [block for block in document.blocks if block.image_description]
        if not described_blocks:
            return problems # No descriptions to associate.

        for problem in problems:
            self._drop_unmentioned_figure_references(problem)
            meaningful_descriptions = [
                block.image_description
                for block in described_blocks
                if block.image_description and self._is_meaningful_visual_context(block.image_description, problem, document)
            ]
            if not meaningful_descriptions:
                continue

            # Prefer explicit figure references, but also attach available visual
            # descriptions when a single visual-dependent problem is present. This
            # avoids dropping diagrams that are adjacent to, but not named by, the
            # OCR text.
            should_attach = bool(problem.figure_references)
            if not should_attach and len(problems) == 1 and self._problem_appears_visual_dependent(problem):
                should_attach = True

            if should_attach:
                # For simplicity, we associate all available descriptions. A more advanced
                # implementation could match "Figure 1" to a specific description.
                problem.referenced_figure_descriptions = meaningful_descriptions
                print(f"Associated {len(problem.referenced_figure_descriptions)} descriptions with {problem.problem_id}")
        
        return problems

    def process_selection(
        self,
        user_selection: UserSelection,
        ui_document: UIDocument,
        source_image: Optional[Image.Image] = None,
        visual_context_override: Optional[str] = None,
        remove_visual_context: bool = False,
    ) -> VisionFinalOutput:
        # Find the full problem object based on the user's selection ID.
        selected_problem = next((p for p in ui_document.problems if p.problem_id == user_selection.problem_id), None)
        if not selected_problem:
            raise ValueError(f"Could not find selected problem with ID: {user_selection.problem_id}")

        # Start with the user's potentially edited problem statement.
        final_problem_statement = user_selection.edited_latex
        visual_context_required = (
            self._problem_appears_visual_dependent(selected_problem)
            or bool(selected_problem.figure_references)
        )

        # If we have stored descriptions for this problem, append them.
        # if selected_problem.referenced_figure_descriptions:
        #     print(f"[DEBUG-3] HANDOFF: Found {len(selected_problem.referenced_figure_descriptions)} descriptions for {selected_problem.problem_id}.")
        #     descriptions_text = "\n\n".join(selected_problem.referenced_figure_descriptions)
        #     final_problem_statement += f"\n\n[Associated Visual Information]:\n{descriptions_text}"
        # # === END KEY LOGIC ===
        
        # The VLM call can now be a secondary, more intelligent step.
        # For now, we pass the text-based context we already have.
        # A full VLM analysis might not even be necessary if the text description is good.
        #visual_context = None 
        # visual_context = self.visual_contextualizer.analyze(...) # You can still run this if you need more than text

        original_descriptions_text = "\n\n".join(selected_problem.referenced_figure_descriptions)
        descriptions_text = (visual_context_override if visual_context_override is not None else original_descriptions_text).strip()
        visual_context_source = (
            "user_override"
            if visual_context_override is not None and descriptions_text
            else "referenced_figure_descriptions"
            if original_descriptions_text
            else None
        )

        if remove_visual_context:
            descriptions_text = ""
            visual_context_source = "user_removed"

        if descriptions_text:

            # Don't embed in problem statement
            final_problem_statement = user_selection.edited_latex

            # Create proper VisualContext object
            visual_context = VisualContext(
                elements=[],
                summary=descriptions_text,
                contains_essential_info=True
            )
        else:
            final_problem_statement = user_selection.edited_latex
            visual_context = None

        return VisionFinalOutput(
            problem_statement=final_problem_statement,
            visual_context=visual_context,
            source_metadata={
                "problem_id": user_selection.problem_id,
                "problem_type": selected_problem.problem_type.value,
                "figure_references": selected_problem.figure_references,
                "processing_method": "marker_then_semantic_grouping",
                "total_available_blocks": len(ui_document.blocks),
                "total_problems_found": len(ui_document.problems),
                "visual_context_required": visual_context_required,
                "visual_context_attached": visual_context is not None,
                "visual_context_description_count": 1 if visual_context is not None else 0,
                "visual_context_source": visual_context_source,
                "visual_context_user_modified": visual_context_override is not None,
                "visual_context_user_removed": remove_visual_context,
                "problem_input_complete": selected_problem.problem_input_complete,
                "missing_problem_content": selected_problem.missing_problem_content,
                "missing_content_reason": selected_problem.missing_content_reason,
                "extraction_recovery_source": selected_problem.extraction_recovery_source,
                "visual_context_missing_reason": (
                    "user_removed"
                    if remove_visual_context
                    else "no_associated_description"
                    if visual_context_required and visual_context is None
                    else None
                ),
                "vlm_analysis_performed": visual_context is not None,
                "document_dimensions": ui_document.dimensions
            }
        )


    def process_document(self, file_input: Union[str, Image.Image]):
        if isinstance(file_input, str):
            # File path provided
            return self.marker_service.convert_document(file_input)
        else:
            # PIL Image provided - need to save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                file_input.save(tmp_file.name)
                try:
                    result = self.marker_service.convert_document(tmp_file.name)
                    return result
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass
