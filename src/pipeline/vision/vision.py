from src.models.manager import ModelManager
from .types import Problem, VisionInput, UserSelection, VisionFinalOutput, UIDocument, VisualContext
from .ui_transformer import UITransformer
from .vlm import VisualContextualizer
from .grouper import SemanticGrouper

from typing import Dict, Union, Optional, List, Tuple
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher
import re

class VisionPipeline:
    def __init__(self, manager: ModelManager):
        self.model_manager = manager
        self.marker_service = manager.marker #access marker service directly
        #self.visual_contextualizer = VisualContextualizer(manager)
        self.grouper = SemanticGrouper(manager)

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

        # Step 4: Grouper identifies distinct problems from the full page text.
        problems = self.grouper.group(ui_document.full_page_text)
        
        # Step 4: Link the found problems back to the original blocks for UI highlighting
        #ui_document.problems = self._link_problems_to_blocks(problems, ui_document)
        problems_with_blocks = self._link_problems_to_blocks(problems, ui_document)

        # Step 5: Explicitly associate figure descriptions with the problems.
        ui_document.problems = self._associate_descriptions_to_problems(problems_with_blocks, ui_document)
        
        return ui_document

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

        # Second pass: Assign blocks to problems
        for problem in problems:
            problem.block_ids = [
                block_id for block_id, (p_id, ratio) in block_assignments.items()
                if p_id == problem.problem_id
            ]
        
        print("--- Block to Problem Linking Complete ---")
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
        return any(marker in text for marker in positive_visual_markers)

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
        source_image: Image.Image,
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
