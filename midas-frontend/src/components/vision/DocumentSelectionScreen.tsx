import React, { useState, useEffect, useRef } from 'react';
import { APIDocument } from '../../types/api';
import { FormattedMathText } from '../ui/FormattedMathText';

interface Props {
  document: APIDocument;
  originalImageBase64: string | null;
  selectedProblemId: string | null;
  editedLatex: string;
  isLoading: boolean;
  error: string | null;
  onSelectProblem: (id: string | null) => void;
  onUpdateLatex: (latex: string) => void;
  onRunPipeline: () => void;
  onBack: () => void;
}

// ── Inline-editable LaTeX field ──────────────────────────────────────────────
// Shows a rendered MathJax preview. Double-click to open a textarea.
// ✓ confirms the edit (returns to preview). ✗ discards the draft.
// ↺ reverts to the original OCR suggestion.
const EditableLatex: React.FC<{
  value: string;         // current committed value (from hook state)
  original: string;      // original OCR suggestion — used for revert
  onChange: (v: string) => void;
  onRevert: () => void;
}> = ({ value, original, onChange, onRevert }) => {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isDirty = value !== original;

  // Keep draft in sync when the committed value changes externally.
  useEffect(() => { setDraft(value); }, [value]);

  const confirm = () => { onChange(draft); setEditing(false); };
  const discard = () => { setDraft(value); setEditing(false); };

  if (editing) {
    return (
      <div style={{ marginTop: 10 }}>
        <textarea
          ref={textareaRef}
          autoFocus
          className="latex-editor"
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Escape') discard();
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) confirm();
          }}
          rows={Math.max(2, Math.ceil(draft.length / 60))}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
          <button
            onClick={confirm}
            title="Confirm (⌘Enter)"
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 28, height: 28, borderRadius: '50%', border: '1.5px solid var(--verified-bd)', background: 'var(--verified-bg)', color: 'var(--verified)', cursor: 'pointer', fontSize: 14, fontWeight: 700 }}
          >
            ✓
          </button>
          <button
            onClick={discard}
            title="Discard changes"
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 28, height: 28, borderRadius: '50%', border: '1.5px solid var(--rule)', background: 'var(--parchment)', color: 'var(--ink-3)', cursor: 'pointer', fontSize: 14 }}
          >
            ✗
          </button>
          {draft !== original && (
            <button
              onClick={() => { setDraft(original); }}
              style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: 'var(--ink-3)', background: 'transparent', border: '1px solid var(--rule)', borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}
            >
              ↺ revert to original
            </button>
          )}
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9.5, color: 'var(--rule)' }}>
            ⌘Enter to confirm · Esc to discard
          </span>
        </div>
      </div>
    );
  }

  // Preview state — rendered LaTeX, double-click to edit
  return (
    <div style={{ marginTop: 10 }}>
      <div
        onDoubleClick={() => { setDraft(value); setEditing(true); }}
        title="Double-click to edit"
        style={{
          padding: '10px 14px',
          borderRadius: 5,
          border: '1.5px solid var(--accent-mid)',
          background: 'rgba(30,58,95,0.04)',
          cursor: 'text',
          minHeight: 36,
        }}
      >
        <FormattedMathText content={value} fontSize={15} lineHeight={1.7} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9.5, color: 'var(--rule)' }}>
          double-click to edit
        </span>
        {isDirty && (
          <button
            onClick={onRevert}
            style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: 'var(--ink-3)', background: 'transparent', border: '1px solid var(--rule)', borderRadius: 3, padding: '2px 8px', cursor: 'pointer' }}
          >
            ↺ revert
          </button>
        )}
        {isDirty && (
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9.5, color: 'var(--amber)' }}>
            modified
          </span>
        )}
      </div>
    </div>
  );
};

// ── Main selection screen ─────────────────────────────────────────────────────
export const DocumentSelectionScreen: React.FC<Props> = ({
  document,
  originalImageBase64,
  selectedProblemId,
  editedLatex,
  isLoading,
  error,
  onSelectProblem,
  onUpdateLatex,
  onRunPipeline,
  onBack,
}) => {
  const hasProblems = document.problems.length > 0;
  const selectedProblem = document.problems.find(p => p.problem_id === selectedProblemId) ?? null;

  const handleRevert = () => {
    if (selectedProblem) onUpdateLatex(selectedProblem.problem_text);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'row', height: '100vh', overflow: 'hidden' }}>

      {/* ── Left: dark document panel ───────────────────────────────────────── */}
      <div style={{ width: 400, flexShrink: 0, background: '#2a2520', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '16px 18px 12px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.35)' }}>
            Document preview
          </div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 18px' }}>
          {originalImageBase64 ? (
            <div style={{ background: 'white', borderRadius: 3, boxShadow: '0 4px 24px rgba(0,0,0,0.4)', overflow: 'hidden' }}>
              <img
                src={`data:image/png;base64,${originalImageBase64}`}
                alt="Document"
                style={{ width: '100%', height: 'auto', display: 'block' }}
              />
            </div>
          ) : (
            <div style={{ color: 'rgba(255,255,255,0.2)', fontFamily: "'JetBrains Mono', monospace", fontSize: 12, textAlign: 'center', marginTop: 40 }}>
              no preview available
            </div>
          )}
        </div>
      </div>

      {/* ── Right: selection panel ──────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--cream)' }}>

        {/* Header */}
        <div style={{ padding: '22px 32px 18px', borderBottom: '1px solid var(--rule)', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, flexShrink: 0 }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 600, letterSpacing: '0.06em', color: 'var(--ink)' }}>MIDAS</div>
            <div style={{ fontSize: 10.5, fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-3)', marginTop: 1 }}>
              phi4-mini-reasoning
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 10.5, color: 'var(--ink-3)', marginTop: 4 }}>
            {[{ done: true }, { done: false, curr: true }, { done: false }, { done: false }].map((p, i) => (
              <div key={i} style={{ width: 20, height: 4, borderRadius: 2, background: p.done ? 'var(--verified)' : (p as any).curr ? 'var(--accent)' : 'var(--rule)' }} />
            ))}
            <span style={{ marginLeft: 4 }}>Step 2 of 4</span>
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '26px 32px 32px' }} className="thin-scroll">

          {hasProblems ? (
            <>
              <div style={{ fontSize: 9.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 4 }}>
                Problem selection
              </div>
              <div style={{ fontSize: 22, color: 'var(--ink)', marginBottom: 4, lineHeight: 1.3 }}>
                {document.problems.length} problem{document.problems.length !== 1 ? 's' : ''} found
              </div>
              <div style={{ fontSize: 14, fontStyle: 'italic', color: 'var(--ink-3)', fontFamily: "'Crimson Pro', serif", marginBottom: 24 }}>
                Select a problem, then double-click the highlighted expression to correct any OCR errors before analysing.
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {document.problems.map((problem, i) => {
                  const isSelected = problem.problem_id === selectedProblemId;

                  return (
                    <div
                      key={problem.problem_id}
                      style={{
                        border: `1.5px solid ${isSelected ? 'var(--accent)' : 'var(--rule)'}`,
                        borderRadius: 7,
                        padding: '16px 18px',
                        cursor: isSelected ? 'default' : 'pointer',
                        background: isSelected ? 'var(--accent-lt)' : 'var(--cream)',
                        boxShadow: isSelected ? '0 0 0 3px rgba(30,58,95,0.1)' : 'none',
                        transition: 'all 0.16s',
                        display: 'flex',
                        gap: 14,
                        alignItems: 'flex-start',
                      }}
                      onClick={() => { if (!isSelected) onSelectProblem(problem.problem_id); }}
                    >
                      {/* Number bubble */}
                      <div style={{ width: 28, height: 28, borderRadius: '50%', border: `1.5px solid ${isSelected ? 'var(--accent)' : 'var(--rule)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: isSelected ? 'white' : 'var(--ink-3)', flexShrink: 0, marginTop: 2, background: isSelected ? 'var(--accent)' : 'var(--cream)', transition: 'all 0.16s' }}>
                        {i + 1}
                      </div>

                      <div style={{ flex: 1, minWidth: 0 }}>
                        {isSelected ? (
                          // Selected: show editable LaTeX preview (no separate description label)
                          <>
                            <div style={{ fontSize: 9.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span>Problem statement</span>
                              <span style={{ color: 'var(--ink-3)', textTransform: 'none', letterSpacing: 0, fontStyle: 'italic' }}>— will be sent to model</span>
                            </div>
                            <EditableLatex
                              value={editedLatex}
                              original={problem.problem_text}
                              onChange={onUpdateLatex}
                              onRevert={handleRevert}
                            />
                          </>
                        ) : (
                          // Unselected: rendered preview, click to select
                          <div style={{ color: 'var(--ink)' }}>
                            <FormattedMathText content={problem.problem_text} fontSize={15} lineHeight={1.65} />
                          </div>
                        )}
                      </div>

                      {isSelected && (
                        <div style={{ width: 20, height: 20, borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: 10, flexShrink: 0, marginTop: 4 }}>
                          ✓
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 9.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 4 }}>
                Problem selection
              </div>
              <div style={{ fontSize: 22, color: 'var(--ink)', marginBottom: 4, lineHeight: 1.3 }}>
                No problems detected
              </div>
              <div style={{ fontSize: 14, fontStyle: 'italic', color: 'var(--ink-3)', fontFamily: "'Crimson Pro', serif", marginBottom: 24 }}>
                MIDAS couldn't identify a clear mathematical problem. Try a clearer photograph.
              </div>
              <div style={{ border: '1.5px dashed var(--rule)', borderRadius: 7, padding: '32px 24px', textAlign: 'center', background: 'var(--parchment)' }}>
                <div style={{ fontSize: 32, color: 'var(--rule)', marginBottom: 10 }}>⊘</div>
                <div style={{ fontSize: 17, color: 'var(--ink-2)', marginBottom: 6 }}>Nothing found in the document</div>
                <div style={{ fontSize: 14, fontStyle: 'italic', color: 'var(--ink-3)', fontFamily: "'Crimson Pro', serif" }}>
                  The image may be too low-resolution or may not contain a maths problem.
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: '16px 32px 20px', borderTop: '1px solid var(--rule)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0, background: 'var(--parchment)' }}>
          <div style={{ fontSize: 13, fontStyle: 'italic', color: error ? 'var(--failed)' : 'var(--ink-3)', fontFamily: "'Crimson Pro', serif" }}>
            {error ?? (selectedProblemId ? 'Ready to analyse' : 'Select a problem above')}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={onBack}
              style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: '8px 16px', borderRadius: 4, cursor: 'pointer', border: '1px solid var(--rule)', background: 'transparent', color: 'var(--ink-2)', letterSpacing: '0.02em' }}
            >
              ← Upload different file
            </button>
            <button
              onClick={onRunPipeline}
              disabled={!selectedProblemId || !editedLatex.trim() || isLoading}
              style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, padding: '10px 28px', borderRadius: 5, cursor: (!selectedProblemId || isLoading) ? 'not-allowed' : 'pointer', border: '1px solid var(--accent)', background: (!selectedProblemId || isLoading) ? 'var(--rule)' : 'var(--accent)', color: (!selectedProblemId || isLoading) ? 'var(--ink-3)' : 'white', letterSpacing: '0.04em' }}
            >
              {isLoading ? 'Processing…' : 'Analyse selected →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
