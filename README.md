# MIDAS

**Mathematical Intelligence with Deductive, Algebraic Synthesis**

MIDAS is a dissertation system that takes a photograph of a handwritten or printed mathematics problem, solves it using a structured proof-style reasoning model, checks every step of that proof symbolically with SymPy, and generates targeted student feedback for any step that fails verification. The complete attempt — including any repair cycles — is logged to a JSONL file in a format suitable for reward-model training.

The central argument of the system is that symbolic verification and structured proof-style reasoning can be introduced into math education tools, and that the feedback loop between attempt, verify, and repair produces training signal for learning-to-prove systems.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Pipeline walkthrough](#pipeline-walkthrough)
   - [Stage 1 — Vision](#stage-1--vision)
   - [Stage 2 — Reasoning](#stage-2--reasoning)
   - [Stage 3 — Verification](#stage-3--verification)
   - [Stage 4 — Feedback generation](#stage-4--feedback-generation)
   - [Stage 5 — Repair loop](#stage-5--repair-loop)
   - [Stage 6 — Trajectory logging](#stage-6--trajectory-logging)
3. [Models and configuration](#models-and-configuration)
4. [Prompt system](#prompt-system)
5. [Data structures](#data-structures)
6. [API reference](#api-reference)
7. [Frontend](#frontend)
8. [Running the system](#running-the-system)
9. [Project structure](#project-structure)

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (React)                       │
│   Upload → Select problem → Loading → Results               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP  /api/v1/
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI (Python)                          │
│  /vision   /reasoning   /verification   /trajectories       │
└──────┬──────────┬──────────────┬──────────────┬────────────┘
       │          │              │              │
  ┌────▼────┐ ┌──▼────────┐ ┌──▼──────────┐  ┌▼───────────┐
  │  Vision  │ │ Reasoning │ │Verification │  │ Trajectory │
  │ Pipeline │ │ Pipeline  │ │  Pipeline   │  │   Logger   │
  └────┬────┘ └──┬────────┘ └──┬──────────┘  └────────────┘
       │          │              │
  ┌────▼──────────▼──────────────▼──────────────────────────┐
  │                     Model Manager                        │
  │   Prompt rendering (Jinja2) → Ollama / OpenRouter        │
  └──────────────────────────────────────────────────────────┘
```

All model calls go through a single `ModelManager` that reads `src/config/config.yaml`, renders the appropriate versioned Jinja2 prompt template, and dispatches to an Ollama or OpenAI-compatible provider. This makes it straightforward to swap any model or prompt version without touching pipeline code.

---

## Pipeline walkthrough

### Stage 1 — Vision

**Entry point:** `POST /api/v1/vision/upload`  
**Source:** `src/pipeline/vision/`

The user uploads a PNG, JPEG, or PDF. Before any OCR runs, a lightweight VLM call (`qwen2.5vl:7b`, `vision/validate@v1`) checks whether the image actually contains mathematics; non-math images are rejected immediately with a 400.

Accepted images are normalised — transparent PNGs are composited onto a white background so that OCR and the LLM description step see clean black-on-white text — then passed to **Marker PDF** for document layout analysis and OCR. Marker is configured with `use_llm: true` so that figure and picture blocks are described by Gemini 1.5 Flash rather than silently dropped.

The raw Marker output is transformed by `UITransformer` into a flat list of `UIBlock` objects, each with its type, bounding polygon, raw text content, and an optional LLM-generated image description. The full page text is assembled from these blocks and sent to `SemanticGrouper` (`qwen3:8b`, `vision/group_problems@v3`), which identifies the distinct mathematical problems on the page as a structured JSON list. Each identified problem is tagged with a heuristic problem type (`algebra`, `calculus`, `proof`, `geometry`, `statistics`, `number_theory`, `linear_algebra`, or `other`) by a keyword classifier. The problems are then linked back to their originating blocks via fuzzy text matching so the frontend can highlight them in context.

The session (document, image, block data) is stored server-side keyed by a UUID that the frontend holds for the duration of the interaction.

---

### Stage 2 — Reasoning

**Entry point:** `POST /api/v1/vision/complete` (full pipeline) or `POST /api/v1/reasoning/reason` (standalone)  
**Source:** `src/pipeline/reasoning/`

After the user selects a problem and optionally corrects the OCR-extracted LaTeX, the problem statement is sent to `ReasoningPipeline`, which calls `phi4-mini-reasoning` with the `reasoning/solve@v2` prompt.

The v2 prompt requires the model to output a structured XML response:

```xml
<think>...exploratory scratchpad...</think>

<solution>
  <given>Problem restated symbolically</given>

  <step number="1">
    <claim>A single precise mathematical assertion</claim>
    <latex>$expression$</latex>
    <justification>Rule or operation used</justification>
  </step>

  ...

  <answer>
    <value>Final answer in plain text</value>
    <latex>$LaTeX form$</latex>
  </answer>
</solution>
```

The parser extracts each `<step>` into a `ReasoningStep` dataclass with fields `step_number`, `claim`, `latex_expression`, and `justification`. The `verification_status` and `verification_note` fields are initially `None` — they are filled in during Stage 3.

If the model does not produce the structured format (which can happen with some prompts or quantised models), a fallback parser wraps the raw response in a single-step `ReasoningOutput` so the rest of the pipeline can continue.

**Key type:**
```python
@dataclass
class ReasoningStep:
    step_number: int
    claim: str
    justification: str
    latex_expression: Optional[str] = None
    verification_status: Optional[bool] = None  # filled by verification
    verification_note: Optional[str] = None      # SymPy finding on failure
    feedback: Optional[str] = None               # student-facing explanation
```

---

### Stage 3 — Verification

**Entry point:** called internally by `VerificationOrchestrator`  
**Source:** `src/pipeline/verification/`

Verification follows a strict **Generate → Execute → Analyse** contract. The `VerificationOrchestrator` manages the outer retry loop; `VerificationPipeline` handles one attempt.

**Code generation.** `SymPyCodeGenerator` calls `qwen2.5-coder:7b-instruct` with the `codegen/baseline_codegen@v6` prompt. The prompt receives the full structured `ReasoningOutput` — crucially including the typed step list — and requires the model to emit exactly one JSON line per step:

```json
{"step": 1, "description": "claim text", "verified": true,  "note": "SymPy confirms"}
{"step": 2, "description": "claim text", "verified": false, "note": "SymPy gives x=7, not x=4"}
{"final_answer_verified": true, "answer": "x=3", "note": ""}
```

This is the key alignment: because the model receives `step_number` as a field on each `ReasoningStep` and must key its output by that number, there is a guaranteed one-to-one correspondence between reasoning steps and verification results. Previous versions guessed the step correspondence positionally; v6 makes it explicit.

**Sandboxed execution.** The generated Python code runs in `SafeExecutor`, which applies a wall-clock timeout and memory limit. Execution captures stdout and stderr separately.

**Output parsing.** `VerificationOutputParser` reads the stdout line by line, parses JSON objects keyed by `step`, deduplicates by step number, and sorts by step number. If stdout is empty or no step JSON lines are found, it returns an error string rather than an exception — this distinguishes a contract violation (codegen fault) from a math failure (reasoning fault).

**Result annotation.** `_annotate_reasoning_steps()` writes `verification_status` (True/False) and `verification_note` back onto the originating `ReasoningStep` objects in-place. This is what allows the frontend to show per-step ✓/✗ indicators without any additional data transformation.

**Fault classification.** The pipeline distinguishes two fault types:

- **Reasoning fault** (`status: "failed_reasoning"`) — the code executed cleanly but SymPy disproved one or more steps. The math is wrong. This triggers the repair loop.
- **Codegen fault** (`status: "failed_codegen"`) — the code crashed, timed out, or violated the JSON contract. The math may be correct but the verifier is broken. One repair attempt is made by injecting the error message back into the prompt.

---

### Stage 4 — Feedback generation

**Source:** `src/pipeline/reasoning/feedback.py`

After annotation, `FeedbackGenerator.annotate_failed_steps()` iterates over every step with `verification_status=False` and calls the reasoning model with the `reasoning/student_feedback@v1` prompt. The prompt receives the original problem statement, the failed step's claim and LaTeX, the justification the student gave, and the SymPy finding. The model returns 2–4 sentences of targeted feedback:

- **Specific**: names the exact wrong claim
- **Correct**: states what is actually true
- **Explanatory**: identifies the mistake type (sign error, wrong formula, incorrect substitution)
- **Actionable**: ends with what to fix

This feedback is stored on `ReasoningStep.feedback` and is included in the API response for immediate display. The frontend also exposes an on-demand `/api/v1/reasoning/feedback` endpoint for the case where a user clicks a failed step that doesn't have pre-generated feedback.

---

### Stage 5 — Repair loop

**Source:** `src/pipeline/verification/verification_orchestrator.py`

If the initial verification returns `status: "failed_reasoning"`, the orchestrator attempts a repair. It calls `ReasoningPipeline` again using the `reasoning/repair@v1` prompt, injecting:

- The original problem statement
- The `worked_solution` from the failed attempt
- A concise error summary listing which steps failed and what SymPy found

The repaired `ReasoningOutput` is then re-verified from scratch. This loop runs up to `max_reasoning_attempts` times (default 2). The system stops early if verification succeeds or if the fault type changes to a non-reasoning error.

The repair loop is implemented with a single exit point: `close_trajectory()` is always called regardless of which branch exits, ensuring no trajectories are left open on success or failure.

---

### Stage 6 — Trajectory logging

**Source:** `src/pipeline/trajectory.py`

Every call to `VerificationOrchestrator.verify_with_repair()` produces one JSONL record at `trajectories/midas_trajectories.jsonl`. The record captures the full repair trajectory:

```json
{
  "trajectory_id": "a3f7c81...",
  "timestamp": "2026-05-05T14:30:00Z",
  "problem": {
    "statement": "Solve for x: 3x - 9 = 0",
    "type": "algebra",
    "source": ""
  },
  "attempts": [
    {
      "attempt_number": 1,
      "reasoning": {
        "steps": [
          {"step_number": 1, "claim": "...", "verification_status": false, "verification_note": "...", "feedback": "..."}
        ],
        "final_answer": "3"
      },
      "verification": {
        "status": "failed_reasoning",
        "steps_verified": 2,
        "steps_failed": 1,
        "errors": [{"error_type": "ASSERTION_FAILED", "message": "..."}]
      },
      "generated_code": "import sympy as sp\n..."
    },
    {
      "attempt_number": 2,
      "reasoning": { "...repaired steps..." },
      "verification": {"status": "verified", "steps_verified": 3, "steps_failed": 0, "errors": []}
    }
  ],
  "outcome": {
    "final_status": "verified",
    "attempt_count": 2,
    "difficulty_signal": 0.33
  }
}
```

The **difficulty signal** is computed as `(attempt_count - 1) / max_attempts`. A problem solved on the first try scores 0.0; one that exhausted all attempts scores 1.0. This provides a continuous reward signal for reward-model training without requiring human labelling — the system generates it automatically from the repair loop dynamics.

Aggregate statistics are available at `GET /api/v1/trajectories/stats`.

---

## Models and configuration

All models run locally via **Ollama**. No internet connection is required at inference time.

| Task | Model | Role |
|---|---|---|
| Document OCR | `marker-pdf` + Gemini 1.5 Flash | Layout extraction and image description |
| Math validation | `qwen2.5vl:7b` | Rejects non-mathematical uploads |
| Problem grouping | `qwen3:8b` | Semantic extraction of distinct problems from OCR text |
| Reasoning | `phi4-mini-reasoning:latest` | Structured proof-style solution generation |
| Reasoning repair | `phi4-mini-reasoning:latest` | Re-attempts failed proofs with SymPy error feedback |
| Student feedback | `phi4-mini-reasoning:latest` | 2–4 sentence targeted explanation for failed steps |
| Step explanation | `qwen3:8b` | On-demand elaboration of any step |
| Verification codegen | `qwen2.5-coder:7b-instruct` | Generates step-aligned SymPy verification code |
| Symbolic execution | SymPy (no LLM) | Evaluates mathematical correctness |

Configuration lives in `src/config/config.yaml`. Each task entry specifies a provider, model, parameters, and a `prompt_ref` in the form `category/name@version`. Changing which prompt a task uses requires only a one-line edit to the config; the old prompt directory is preserved as a fallback.

---

## Prompt system

Prompts are versioned Jinja2 templates stored in `prompts/<category>/<name>/<version>/`. Each version directory contains three files:

| File | Purpose |
|---|---|
| `system.j2` | System message template |
| `user.j2` | User message template |
| `config.yaml` | Metadata (task, version, description) |

The `PromptManager` compiles templates with `jinja2.Environment(undefined=StrictUndefined)` — any missing variable is a hard error, not a silent empty string. Templates receive the full Python object as context, so `{{ reasoning.steps }}` iterates over the actual `List[ReasoningStep]` without serialisation boilerplate.

**Active prompt versions:**

| Prompt ref | Purpose |
|---|---|
| `reasoning/solve@v2` | Structured proof-style solution (XML step format) |
| `reasoning/repair@v1` | Re-solve given SymPy error feedback |
| `reasoning/student_feedback@v1` | Targeted 2–4 sentence step correction |
| `reasoning/explain_step@v1` | On-demand step elaboration |
| `codegen/baseline_codegen@v6` | Step-aligned SymPy code generation |
| `vision/validate@v1` | Binary math content check |
| `vision/group_problems@v3` | Extract problem list from OCR text |

---

## Data structures

### `ReasoningStep`
The central data structure. A single logical step in a proof. Populated progressively as the pipeline runs.

```python
@dataclass
class ReasoningStep:
    step_number: int
    claim: str                        # "The discriminant is 25"
    justification: str                # "by discriminant formula b² − 4ac"
    latex_expression: Optional[str]   # "$\Delta = (-7)^2 - 4(3)(2) = 25$"
    verification_status: Optional[bool]  # None → True / False after SymPy
    verification_note: Optional[str]     # SymPy finding when False
    feedback: Optional[str]              # Student-facing explanation when False
```

### `ReasoningOutput`
Holds the full structured solution. `worked_solution` is a backwards-compatible computed property that reconstructs a plain-text solution string from the step list — any code that predates the typed step list still works.

### `VerificationResult`
The outcome of one verification attempt: status, confidence score, per-step verifications, error list, and the generated SymPy code.

### `TrajectoryRecord`
One logged problem: attempt list, final outcome, difficulty signal. Written as newline-delimited JSON.

---

## API reference

The FastAPI server runs at `http://localhost:8000`. Interactive docs at `/docs`.

### Vision

| Endpoint | Description |
|---|---|
| `POST /api/v1/vision/upload` | Upload image or PDF; returns document with identified problems |
| `POST /api/v1/vision/complete` | Run the full pipeline on a selected problem |

### Reasoning

| Endpoint | Description |
|---|---|
| `POST /api/v1/reasoning/reason` | Standalone reasoning on a problem statement |
| `POST /api/v1/reasoning/feedback` | On-demand student feedback for a single failed step |
| `POST /api/v1/reasoning/explain` | Detailed explanation of any step |

### Trajectories

| Endpoint | Description |
|---|---|
| `GET /api/v1/trajectories/` | Most recent N trajectories (default 20, max 100) |
| `GET /api/v1/trajectories/stats` | Aggregate statistics: total, success rate, mean difficulty, by type |

---

## Frontend

A React/TypeScript single-page app in `midas-frontend/`. The design system uses EB Garamond, Crimson Pro, and JetBrains Mono with a cream/parchment/navy palette.

**Screens:**

**Upload** — Drop a PDF or image onto the dropzone.

**Selection** — After OCR, the document image appears in a dark preview panel. The right panel shows each identified problem as a card. The LaTeX expression on each card is editable inline: clicking the expression activates a textarea; Escape or Enter confirms; a revert button restores the original OCR suggestion if the text has been changed.

**Loading** — A sidebar shows the four pipeline stages (Vision / Reasoning / Verification / Results) with done/active/wait indicators. The main area shows the current stage name, a pipeline track with animated connectors, and a live substep log.

**Results** — A 220px sidebar shows the MIDAS brand, all pipeline stages ticked, the problem statement, and summary metadata. The main column has three tabs:

- **Solution** — Each step rendered as a card with a green ✓ or red ✗ left border. Failed steps show the SymPy finding in monospace and either the pre-generated feedback or a "Get explanation →" button that fetches it on demand. A Proof view toggle switches to a formal "Given / Steps (with ✓/✗) / Therefore" layout.
- **Thinking** — The raw `<think>` scratchpad from the reasoning model.
- **SymPy code** — The generated verification code, syntax-highlighted.

If the problem required repair attempts, a collapsible amber **Repair history** section appears below the steps. It shows a diff between the initial attempt and each subsequent repair — claims that changed between attempts are shown with strikethrough (old) and replacement (new) text.

The right sidebar shows a verification summary widget with per-step dots, type-aware error cards (reasoning faults in red, codegen faults in amber), and trajectory metadata.

---

## Running the system

### Prerequisites

- Python 3.12
- Node.js 18+
- Ollama running locally (`ollama serve`)
- A Gemini API key (used by Marker PDF for image description)

### Pull models

```bash
ollama pull phi4-mini-reasoning:latest
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b-instruct
ollama pull qwen2.5vl:7b
```

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_server.py
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd midas-frontend
npm install
npm start
```

The UI is available at `http://localhost:3000`.

### Combined (start script)

```bash
./start_dev.sh
```

This activates the venv, sets the SSL certificate path for Marker's font download, checks that Ollama is running and the required models are present, starts the backend in the background, and launches the frontend.

### Verify the pipeline

After starting, you can test the reasoning endpoint directly without an image:

```bash
curl -s -X POST http://localhost:8000/api/v1/reasoning/reason \
  -H "Content-Type: application/json" \
  -d '{"problem_statement": "Solve for x: 2x + 6 = 14"}' | python3 -m json.tool
```

The response `data.steps` should contain a list of `ReasoningStep` objects, each with `claim`, `latex_expression`, and `justification` populated.

Trajectory stats (after running a few problems through the UI):

```bash
curl -s http://localhost:8000/api/v1/trajectories/stats | python3 -m json.tool
```

---

## Project structure

```
MIDAS_FINAL/
├── src/
│   ├── api/
│   │   ├── main.py                   FastAPI app, router registration
│   │   ├── models/                   Pydantic request/response schemas
│   │   │   ├── reasoning.py          ReasoningStepResponse, FeedbackRequest, …
│   │   │   ├── verification.py       VerificationResult schema
│   │   │   └── vision.py             DocumentUploadResponse, …
│   │   └── routers/
│   │       ├── vision.py             /upload, /complete
│   │       ├── reasoning.py          /reason, /feedback, /explain
│   │       ├── verification.py       /verify
│   │       └── trajectories.py       /, /stats
│   ├── models/
│   │   ├── manager.py                ModelManager — config loading, dispatch
│   │   ├── prompts.py                PromptManager — Jinja2 template rendering
│   │   └── providers/
│   │       ├── ollama.py             Ollama HTTP client
│   │       └── openai_sdk.py         OpenAI-compatible client
│   └── pipeline/
│       ├── trajectory.py             TrajectoryLogger — JSONL output
│       ├── reasoning/
│       │   ├── types.py              ReasoningStep, ReasoningOutput
│       │   ├── reasoning.py          ReasoningPipeline — v2 parser + fallback
│       │   └── feedback.py           FeedbackGenerator — per-step explanations
│       ├── verification/
│       │   ├── verification_orchestrator.py  Repair loop, trajectory wiring
│       │   ├── verification.py               Generate → Execute → Analyse
│       │   ├── codegen.py                    SymPyCodeGenerator
│       │   ├── executor.py                   SafeExecutor — sandboxed Python
│       │   ├── parser.py                     VerificationOutputParser
│       │   └── verification_types.py         VerificationResult, StepVerification
│       └── vision/
│           ├── vision.py             VisionPipeline — OCR, grouping, linking
│           ├── grouper.py            SemanticGrouper + problem type classifier
│           ├── ui_transformer.py     Marker JSON → UIBlock list
│           └── types.py              UIBlock, Problem, ProblemType, …
├── prompts/
│   ├── reasoning/
│   │   ├── solve/v2/                 Structured XML proof prompt (active)
│   │   ├── repair/v1/                Repair prompt
│   │   ├── student_feedback/v1/      Step correction feedback
│   │   └── explain_step/v1/          Step elaboration
│   ├── codegen/
│   │   └── baseline_codegen/v6/      Step-aligned SymPy codegen (active)
│   └── vision/
│       ├── validate/v1/              Math content check
│       └── group_problems/v3/        Problem extraction from OCR text
├── midas-frontend/
│   └── src/
│       ├── components/
│       │   ├── vision/
│       │   │   ├── FileUpload.tsx              Upload screen
│       │   │   ├── DocumentSelectionScreen.tsx  Selection + inline LaTeX editing
│       │   │   └── DocumentPreview.tsx          Pre-OCR image preview
│       │   ├── results/
│       │   │   ├── PipelineResults.tsx          Results screen with sidebar
│       │   │   ├── StepCard.tsx                 Per-step card with feedback
│       │   │   ├── RepairHistory.tsx             Attempt diff view
│       │   │   └── ProofView.tsx                Formal proof layout
│       │   └── ui/
│       │       ├── PipelineLoading.tsx           Loading screen with stage track
│       │       └── SmartMathRenderer.tsx         MathML/LaTeX renderer
│       ├── hooks/useDocumentState.ts             All app state and actions
│       ├── services/SimpleAPIService.ts          API client
│       └── types/api.ts                         TypeScript response types
├── tests/
│   └── verification/
│       └── test_step_alignment.py               Step annotation + parser tests
├── trajectories/                                JSONL logs (gitignored)
├── src/config/config.yaml                       Model and task configuration
├── requirements.in                              Direct Python dependencies
├── requirements.txt                             Pinned lockfile (pip-compile)
└── run_server.py                                Development server entry point
```
