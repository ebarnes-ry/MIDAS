"""
Vision pipeline API endpoints.
"""

import base64
import io
import os
import tempfile
import time
from typing import Optional

from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from PIL import Image

from src.api.limiting import limiter
from src.api.quota import (
    require_explain_quota,
    require_pipeline_quota,
    require_upload_quota,
)
from src.models.manager import ModelManager
from src.pipeline.reasoning.reasoning import ReasoningContractError, ReasoningPipeline
from src.pipeline.reasoning.types import ReasoningInput, ReasoningOutput
from src.pipeline.trajectory import TrajectoryLogger
from src.pipeline.verification.verification_orchestrator import VerificationOrchestrator
from src.pipeline.verification.verification_types import (
    ErrorType,
    VerificationError,
    VerificationResult,
)
from src.pipeline.vision.types import (
    MathValidationResult,
    UIDocument,
    UIBlock,
    UserSelection,
    VisionInput,
)
from src.pipeline.vision.preloaded_examples import (
    available_preloaded_examples,
    load_preloaded_example,
)
from src.pipeline.vision.vision import VisionPipeline

from ..dependencies.session import (
    SessionManager,
    get_model_manager,
    get_session_manager,
)
from ..models.reasoning import (
    ReasoningExplainData,
    ReasoningExplainRequest,
    ReasoningExplainResponse,
)
from ..models.vision import (
    APIDocument,
    APIBlock,
    APIProblem,
    DocumentUploadData,
    DocumentUploadResponse,
    UserSelectionRequest,
)

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_UPLOAD_TYPES = {"image/png", "image/jpeg"}


def _extract_image_description(html: Optional[str]) -> Optional[str]:
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    img_desc_tag = soup.find("p", attrs={"role": "img"})

    if not img_desc_tag:
        return None

    desc_text = img_desc_tag.get_text(strip=True)
    prefix = "image description:"

    if desc_text.lower().startswith(prefix):
        return desc_text[len(prefix):].strip()

    return desc_text


def _extract_and_crop_image_region(
    block: UIBlock,
    original_image: Image.Image,
) -> Optional[str]:
    if not original_image or not block.polygon:
        return None

    try:
        if isinstance(block.polygon[0], (list, tuple)):
            flat_polygon = [coord for point in block.polygon for coord in point]
        else:
            flat_polygon = block.polygon

        if len(flat_polygon) < 8:
            return None

        x_coords = flat_polygon[0::2]
        y_coords = flat_polygon[1::2]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        img_width, img_height = original_image.size
        bounds = (
            max(0, int(x_min)),
            max(0, int(y_min)),
            min(img_width, int(x_max)),
            min(img_height, int(y_max)),
        )

        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return None

        cropped_image = original_image.crop(bounds)

        buffer = io.BytesIO()
        cropped_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Error extracting image region for block {block.id}: {e}")
        return None


def convert_ui_block_to_api_block(
    ui_block: UIBlock,
    original_image: Image.Image,
    include_cropped_images: bool = True,
) -> APIBlock:
    cropped_b64 = None

    if (
        include_cropped_images
        and
        ui_block.block_type.lower() in {"figure", "picture", "table", "diagram"}
        and not ui_block.is_editable
    ):
        cropped_b64 = _extract_and_crop_image_region(ui_block, original_image)

    return APIBlock(
        id=ui_block.id,
        block_type=ui_block.block_type,
        html=ui_block.html,
        polygon=ui_block.polygon,
        bbox=ui_block.bbox,
        is_editable=ui_block.is_editable,
        latex_content=ui_block.latex_content,
        image_description=ui_block.image_description
        or _extract_image_description(ui_block.html),
        cropped_image=cropped_b64,
    )


def convert_ui_document_to_api_document(
    ui_document: UIDocument,
    original_image: Image.Image,
    include_cropped_images: bool = True,
) -> APIDocument:
    api_blocks = [
        convert_ui_block_to_api_block(block, original_image, include_cropped_images)
        for block in ui_document.blocks
    ]

    api_problems = [
        APIProblem(
            problem_id=p.problem_id,
            problem_text=p.problem_text,
            block_ids=p.block_ids,
            problem_type=p.problem_type.value,
            figure_references=p.figure_references,
            visual_context_required=(
                p.problem_type.value == "geometry"
                or bool(p.figure_references)
                or any(
                    marker in p.problem_text.lower()
                    for marker in (
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
                )
            ),
            visual_context_attached=bool(p.referenced_figure_descriptions),
            visual_context_summary=(
                "\n\n".join(p.referenced_figure_descriptions)
                if p.referenced_figure_descriptions
                else None
            ),
            visual_context_description_count=len(p.referenced_figure_descriptions),
            problem_input_complete=p.problem_input_complete,
            missing_problem_content=p.missing_problem_content,
            missing_content_reason=p.missing_content_reason,
            extraction_recovery_source=p.extraction_recovery_source,
        )
        for p in ui_document.problems
    ]

    return APIDocument(blocks=api_blocks, problems=api_problems)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_document_upload_response(
    *,
    document_id: str,
    ui_document: UIDocument,
    original_image_base64: str,
    original_image: Image.Image,
    processing_time: float,
    processing_metadata: dict,
    message: str,
    quota=None,
    include_cropped_images: bool = True,
) -> DocumentUploadResponse:
    api_document = convert_ui_document_to_api_document(
        ui_document,
        original_image,
        include_cropped_images=include_cropped_images,
    )

    response = DocumentUploadResponse(
        success=True,
        message=message,
        data=DocumentUploadData(
            document_id=document_id,
            document=api_document,
            original_image_base64=original_image_base64,
            processing_time=processing_time,
            processing_metadata=processing_metadata,
        ),
    )

    if quota is not None:
        response.data.processing_metadata["quota"] = quota

    return response


def _normalize_uploaded_image(file_content: bytes) -> Image.Image:
    try:
        raw = Image.open(io.BytesIO(file_content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    if raw.mode in ("RGBA", "LA", "P"):
        if raw.mode == "P":
            raw = raw.convert("RGBA")

        background = Image.new("RGB", raw.size, (255, 255, 255))
        alpha = raw.split()[-1]
        background.paste(raw, mask=alpha)
        return background

    return raw.convert("RGB")


@router.get("/examples")
async def list_examples():
    return {"examples": available_preloaded_examples()}


@router.post("/examples/{example_id}", response_model=DocumentUploadResponse)
@limiter.limit("30/minute")
async def load_cached_example(
    request: Request,
    example_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    try:
        payload = load_preloaded_example(example_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown example input.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Cached example result has not been generated yet.",
        ) from exc

    ui_document = payload["ui_document"]
    original_image_base64 = payload["original_image_base64"]
    processing_metadata = dict(payload.get("processing_metadata") or {})
    processing_metadata.update(
        {
            "cached": True,
            "source": "preloaded_example",
            "example_id": example_id,
            "example_label": payload.get("label"),
            "filename": payload.get("file_name"),
        }
    )

    image_data = base64.b64decode(original_image_base64)
    original_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    document_id = session_manager.create_session(
        ui_document=ui_document,
        original_image_base64=original_image_base64,
        processing_metadata=processing_metadata,
    )

    return _build_document_upload_response(
        document_id=document_id,
        ui_document=ui_document,
        original_image_base64=original_image_base64,
        original_image=original_image,
        processing_time=0.0,
        processing_metadata=processing_metadata,
        message="Loaded precomputed example document",
        include_cropped_images=False,
    )


@router.post("/upload", response_model=DocumentUploadResponse)
@limiter.limit("6/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    quota=Depends(require_upload_quota),
    session_manager: SessionManager = Depends(get_session_manager),
    model_manager: ModelManager = Depends(get_model_manager),
):
    start_time = time.time()

    if file.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload must be a PNG or JPEG for this public demo.",
        )

    tmp_filepath: Optional[str] = None

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(file_content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File is too large for the public demo.",
            )

        original_image = _normalize_uploaded_image(file_content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            original_image.save(tmp_file, format="PNG")
            tmp_filepath = tmp_file.name

        print("--- Validating image content ---")
        validation_response = model_manager.call(
            task="validation",
            prompt_ref="vision/validate@v1",
            variables={},
            schema=MathValidationResult,
            images=[original_image],
        )

        if (
            not validation_response.parsed
            or not validation_response.parsed.contains_math
        ):
            reason = (
                validation_response.parsed.reason
                if validation_response.parsed
                else "Could not determine content."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Validation Failed: {reason}",
            )

        print(f"Content validated: {validation_response.parsed.reason}")

        vision_pipeline = VisionPipeline(model_manager)
        vision_input = VisionInput(
            file_path=tmp_filepath,
            file_type=file.content_type,
        )
        ui_document = vision_pipeline.process_input(vision_input)

        original_image_base64 = image_to_base64(original_image)
        processing_time = time.time() - start_time

        processing_metadata = {
            "filename": file.filename,
            "processing_time": processing_time,
            "cached": False,
            "source": "fresh_upload",
            "notice": (
                "Fresh uploads run OCR, layout analysis, and grouping live. "
                "Processing may take several minutes."
            ),
        }

        document_id = session_manager.create_session(
            ui_document=ui_document,
            original_image_base64=original_image_base64,
            processing_metadata=processing_metadata,
        )

        return _build_document_upload_response(
            document_id=document_id,
            ui_document=ui_document,
            original_image_base64=original_image_base64,
            original_image=original_image,
            processing_time=processing_time,
            processing_metadata=processing_metadata,
            message="Document processed successfully",
            quota=quota,
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        if tmp_filepath:
            try:
                os.unlink(tmp_filepath)
            except FileNotFoundError:
                pass


@router.post("/complete")
@limiter.limit("2/minute;10/hour")
async def complete_pipeline(
    request: Request,
    body: UserSelectionRequest,
    quota=Depends(require_pipeline_quota),
    session_manager: SessionManager = Depends(get_session_manager),
    model_manager: ModelManager = Depends(get_model_manager),
):
    full_start_time = time.time()

    session = session_manager.get_session(body.document_id)
    if not session:
        raise HTTPException(status_code=404, detail="Document not found or session expired")

    try:
        vision_start_time = time.time()

        user_selection = UserSelection(
            problem_id=body.problem_id,
            edited_latex=body.edited_latex,
            original_image_path="",
        )

        image_data = base64.b64decode(session.original_image_base64)
        source_image = Image.open(io.BytesIO(image_data))

        vision_pipeline = VisionPipeline(model_manager)
        vision_output = vision_pipeline.process_selection(
            user_selection,
            session.ui_document,
            source_image,
            visual_context_override=body.visual_context_override,
            remove_visual_context=body.remove_visual_context,
        )

        vision_time = time.time() - vision_start_time
        reasoning_start_time = time.time()

        reasoning_pipeline = ReasoningPipeline(model_manager)
        reasoning_input = ReasoningInput(
            problem_statement=vision_output.problem_statement,
            visual_context=(
                vision_output.visual_context.summary
                if vision_output.visual_context
                else None
            ),
            source_metadata=vision_output.source_metadata,
        )

        try:
            reasoning_output = reasoning_pipeline.process(reasoning_input)

        except ReasoningContractError as e:
            reasoning_time = time.time() - reasoning_start_time
            total_processing_time = time.time() - full_start_time

            synthetic_reasoning = ReasoningOutput(
                original_problem=vision_output.problem_statement,
                steps=[],
                final_answer="",
                think_reasoning="",
                processing_metadata={
                    "status": "failed_contract",
                    "error_type": "contract_violation",
                    "error": e.message,
                    "prompt_ref": e.prompt_ref,
                    "model": e.model,
                    "raw_response_length": len(e.raw_output or ""),
                },
            )

            contract_result = VerificationResult(
                status="failed_contract",
                confidence_score=0.0,
                reasoning_output=synthetic_reasoning,
                generated_code="",
                answer_match=None,
                errors=[
                    VerificationError(
                        error_type=ErrorType.CONTRACT_VIOLATION,
                        message=e.message,
                    )
                ],
                metadata={
                    "contract_failure": True,
                    "contract_source": "reasoning_solve",
                    "prompt_ref": e.prompt_ref,
                    "model": e.model,
                    "raw_response_length": len(e.raw_output or ""),
                },
            )

            try:
                logger = TrajectoryLogger(
                    log_path=model_manager.config.get(
                        "trajectory_log_path",
                        "trajectories/midas_trajectories.jsonl",
                    )
                )
                tid = logger.start_trajectory(
                    problem_statement=vision_output.problem_statement,
                    problem_type=str(
                        vision_output.source_metadata.get("problem_type", "unknown")
                    ),
                    problem_source="vision_complete",
                )
                logger.log_attempt(
                    trajectory_id=tid,
                    attempt_number=1,
                    reasoning_output=synthetic_reasoning,
                    verification_result=contract_result,
                    generated_code="",
                )
                logger.close_trajectory(
                    trajectory_id=tid,
                    final_status="failed_contract",
                    max_attempts=1,
                )
            except Exception as log_error:
                print(f"Could not log reasoning contract failure trajectory: {log_error}")

            contract_metadata = {
                "contract_failure": True,
                "contract_source": "reasoning_solve",
                "prompt_ref": e.prompt_ref,
                "model": e.model,
                "raw_response_length": len(e.raw_output or ""),
            }

            response_data = {
                "vision": {
                    "problem_statement": vision_output.problem_statement,
                    "visual_context": vision_output.visual_context,
                    "processing_time": vision_time,
                    "metadata": vision_output.source_metadata,
                },
                "reasoning": {
                    "original_problem": vision_output.problem_statement,
                    "steps": [],
                    "worked_solution": "",
                    "final_answer": "",
                    "think_reasoning": "",
                    "processing_time": reasoning_time,
                    "metadata": {
                        "status": "failed_contract",
                        "error_type": "contract_violation",
                        "error": e.message,
                        **contract_metadata,
                    },
                },
                "verification": {
                    "original_problem": vision_output.problem_statement,
                    "worked_solution": "",
                    "final_answer": "",
                    "generated_code": "",
                    "processing_time": 0.0,
                    "status": "failed_contract",
                    "confidence_score": 0.0,
                    "answer_match": None,
                    "errors": [
                        {
                            "error_type": "contract_violation",
                            "message": e.message,
                            "line_number": None,
                            "problematic_code": None,
                            "suggested_fix": None,
                            "traceback": None,
                        }
                    ],
                    "repair_history": [],
                    "step_verifications": [],
                    "metadata": {
                        "reasoning_repair_attempts": 0,
                        "codegen_repair_attempts": 0,
                        **contract_metadata,
                    },
                },
                "total_processing_time": total_processing_time,
            }

            return {
                "success": False,
                "message": "Reasoning output violated the structured contract",
                "data": response_data,
                "quota": quota,
            }

        reasoning_time = time.time() - reasoning_start_time
        verification_start_time = time.time()

        verification_orchestrator = VerificationOrchestrator(model_manager)
        verification_result, repair_history = verification_orchestrator.verify_with_repair(
            reasoning_output=reasoning_output,
            max_reasoning_attempts=2,
        )

        verification_time = time.time() - verification_start_time
        total_processing_time = time.time() - full_start_time
        final_reasoning = verification_result.reasoning_output

        def _step_dict(s):
            return {
                "step_number": s.step_number,
                "claim": s.claim,
                "justification": s.justification,
                "latex_expression": s.latex_expression,
                "verification_status": s.verification_status,
                "verification_note": s.verification_note,
                "feedback": getattr(s, "feedback", None),
            }

        repair_attempts_with_steps = [
            {
                "attempt_number": 1,
                "reasoning_steps": [_step_dict(s) for s in reasoning_output.steps],
                "final_answer": reasoning_output.final_answer,
                "verification_status": "initial",
                "steps_verified": 0,
                "steps_failed": 0,
            }
        ]

        for i, r in enumerate(repair_history):
            steps = r.repaired_reasoning.steps if r.repaired_reasoning else []
            repair_attempts_with_steps.append(
                {
                    "attempt_number": i + 2,
                    "reasoning_steps": [_step_dict(s) for s in steps],
                    "final_answer": (
                        r.repaired_reasoning.final_answer
                        if r.repaired_reasoning
                        else ""
                    ),
                    "verification_status": "verified" if r.success else "failed",
                    "steps_verified": 0,
                    "steps_failed": 0,
                }
            )

        response_data = {
            "vision": {
                "problem_statement": vision_output.problem_statement,
                "visual_context": vision_output.visual_context,
                "processing_time": vision_time,
                "metadata": vision_output.source_metadata,
            },
            "reasoning": {
                "original_problem": final_reasoning.original_problem,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "claim": s.claim,
                        "justification": s.justification,
                        "latex_expression": s.latex_expression,
                        "verification_status": s.verification_status,
                        "verification_note": s.verification_note,
                        "feedback": s.feedback,
                    }
                    for s in final_reasoning.steps
                ],
                "worked_solution": final_reasoning.worked_solution,
                "final_answer": final_reasoning.final_answer,
                "think_reasoning": final_reasoning.think_reasoning,
                "processing_time": reasoning_time,
                "metadata": final_reasoning.processing_metadata,
            },
            "verification": {
                "original_problem": final_reasoning.original_problem,
                "worked_solution": final_reasoning.worked_solution,
                "final_answer": final_reasoning.final_answer,
                "generated_code": verification_result.generated_code,
                "processing_time": verification_time,
                "status": verification_result.status,
                "confidence_score": verification_result.confidence_score,
                "answer_match": verification_result.answer_match,
                "errors": [err.model_dump() for err in verification_result.errors],
                "repair_history": repair_attempts_with_steps,
                "step_verifications": [
                    {
                        "step_number": sv.step_number,
                        "description": sv.description,
                        "verified": sv.verified,
                        "note": sv.note,
                    }
                    for sv in verification_result.step_verifications
                ],
                "metadata": {
                    "reasoning_repair_attempts": len(
                        [r for r in repair_history if r.repair_type == "reasoning"]
                    ),
                    "codegen_repair_attempts": len(
                        [r for r in repair_history if r.repair_type == "codegen"]
                    ),
                    **verification_result.metadata,
                },
            },
            "total_processing_time": total_processing_time,
        }

        return {
            "success": True,
            "message": "Complete pipeline processing successful",
            "data": response_data,
            "quota": quota,
        }

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Complete pipeline failed: {str(e)}",
        ) from e


@router.post("/explain", response_model=ReasoningExplainResponse)
@limiter.limit("10/minute")
async def explain_step(
    request: Request,
    body: ReasoningExplainRequest,
    quota=Depends(require_explain_quota),
    model_manager: ModelManager = Depends(get_model_manager),
):
    start_time = time.time()

    try:
        response = model_manager.call(
            task="explain_step",
            prompt_ref="reasoning/explain_step@v1",
            variables={
                "problem_statement": body.problem_statement,
                "worked_solution": body.worked_solution,
                "step_text": body.step_text,
            },
        )

        processing_time = time.time() - start_time

        return ReasoningExplainResponse(
            success=True,
            message="Explanation generated successfully.",
            data=ReasoningExplainData(
                explanation=response.content,
                processing_time=processing_time,
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {e}",
        ) from e
