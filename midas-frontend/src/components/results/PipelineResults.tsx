import React, { useState } from 'react';
import { CompletePipelineResponse } from '../../types/api';
import { StepCard } from './StepCard';
import { RepairHistory } from './RepairHistory';
import { ProofView } from './ProofView';
import { SmartMathRenderer } from '../ui/SmartMathRenderer';

// ── Error type metadata ───────────────────────────────────────────────────────
const ERR_META: Record<string, { label: string; description: string }> = {
  ANSWER_MISMATCH:    { label: 'Reasoning Fault',      description: 'The mathematical argument contains an error.' },
  ASSERTION_FAILED:   { label: 'Step Failed',           description: 'SymPy disproved a step in the solution.' },
  SYNTAX_ERROR:       { label: 'Code Generation Fault', description: 'The verifier code had a syntax error — not a math error.' },
  TIMEOUT:            { label: 'Timeout',               description: 'Verification exceeded the time limit.' },
  SYMBOLIC_FAILURE:   { label: 'Symbolic Limitation',   description: 'SymPy could not verify this symbolically.' },
  CONTRACT_VIOLATION: { label: 'Contract Violation',    description: 'Verifier did not produce expected output format.' },
  answer_mismatch:    { label: 'Reasoning Fault',      description: 'The mathematical argument contains an error.' },
  assertion_failed:   { label: 'Step Failed',           description: 'SymPy disproved a step in the solution.' },
  syntax_error:       { label: 'Code Generation Fault', description: 'The verifier code had a syntax error — not a math error.' },
  timeout:            { label: 'Timeout',               description: 'Verification exceeded the time limit.' },
  symbolic_failure:   { label: 'Symbolic Limitation',   description: 'SymPy could not verify this symbolically.' },
  contract_violation: { label: 'Contract Violation',    description: 'Verifier did not produce expected output format.' },
};

// ── Shared style tokens ───────────────────────────────────────────────────────
const mono: React.CSSProperties = { fontFamily: "'JetBrains Mono', monospace" };
const serif2: React.CSSProperties = { fontFamily: "'Crimson Pro', serif" };

const DetailsBox: React.FC<{ title?: string; children: React.ReactNode }> = ({ title = 'Details', children }) => (
  <details style={{ marginTop: 10 }}>
    <summary style={{ ...mono, cursor: 'pointer', fontSize: 10.5, color: 'var(--ink-3)', letterSpacing: '0.03em' }}>
      {title}
    </summary>
    <div style={{ marginTop: 8, padding: '9px 10px', border: '1px solid var(--rule-lt)', borderRadius: 4, background: 'var(--cream)', color: 'var(--ink-2)', ...mono, fontSize: 10.5, lineHeight: 1.65, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
      {children}
    </div>
  </details>
);

// ── Sidebar ───────────────────────────────────────────────────────────────────
const Sidebar: React.FC<{
  problemStatement: string;
  steps: number;
  repairAttempts: number;
  status: string;
  confidence: number;
  onStartOver: () => void;
}> = ({ problemStatement, steps, repairAttempts, status, confidence, onStartOver }) => {
  const statusOk = status === 'verified';
  const statusLimited = status === 'unsupported' || status === 'needs_visual_context';
  const statusText = statusOk ? '✓ verified' : statusLimited ? status.replace('_', ' ') : '✗ ' + status.replace('_', ' ');
  return (
    <div style={{ width: 220, flexShrink: 0, background: 'var(--parchment)', borderRight: '1px solid var(--rule)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Brand */}
      <div style={{ padding: '20px 18px 16px', borderBottom: '1px solid var(--rule)' }}>
        <div style={{ fontSize: 19, fontWeight: 600, letterSpacing: '0.06em' }}>MIDAS</div>
        <div style={{ ...mono, fontSize: 10.5, color: 'var(--ink-3)', marginTop: 2 }}>phi4-mini-reasoning</div>
      </div>

      {/* Pipeline (all done) */}
      <div style={{ padding: '14px 18px 6px' }}>
        <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 8 }}>
          Pipeline
        </div>
        {['Vision', 'Reasoning', 'Verification', 'Results'].map((name, i) => (
          <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 8px', borderRadius: 4, marginBottom: 2, background: i === 3 ? 'var(--accent-lt)' : 'transparent' }}>
            <div style={{ width: 20, height: 20, borderRadius: '50%', border: `1.5px solid ${i === 3 ? 'var(--accent)' : 'var(--verified-bd)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, ...mono, flexShrink: 0, background: i === 3 ? 'var(--accent-lt)' : 'var(--verified-bg)', color: i === 3 ? 'var(--accent)' : 'var(--verified)' }}>
              {i === 3 ? '4' : '✓'}
            </div>
            <div style={{ fontSize: 13.5, ...serif2, color: i === 3 ? 'var(--accent)' : 'var(--ink-2)', fontWeight: i === 3 ? 600 : 400 }}>
              {name}
            </div>
          </div>
        ))}
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: 'var(--rule)', margin: '8px 18px' }} />

      {/* Problem snippet */}
      <div style={{ padding: '8px 18px' }}>
        <div style={{ ...mono, fontSize: 9.5, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 6 }}>Problem</div>
        <div style={{ fontSize: 13.5, fontStyle: 'italic', color: 'var(--ink-2)', ...serif2, padding: '4px 8px', lineHeight: 1.5 }}>
          <SmartMathRenderer content={problemStatement.length > 100 ? problemStatement.slice(0, 100) + '…' : problemStatement} />
        </div>
      </div>

      <div style={{ height: 1, background: 'var(--rule)', margin: '8px 18px' }} />

      {/* Metadata */}
      <div style={{ padding: '8px 18px', ...mono, fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 2 }}>
        <div>steps &nbsp;<span style={{ color: 'var(--ink-2)' }}>{steps}</span></div>
        <div>attempts &nbsp;<span style={{ color: 'var(--ink-2)' }}>{repairAttempts + 1}</span></div>
        <div>confidence &nbsp;<span style={{ color: 'var(--ink-2)' }}>{(confidence * 100).toFixed(0)}%</span></div>
        <div style={{ marginTop: 8 }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 8px', borderRadius: 3, fontSize: 10.5, border: '1px solid', background: statusOk ? 'var(--verified-bg)' : statusLimited ? 'var(--amber-bg)' : 'var(--failed-bg)', borderColor: statusOk ? 'var(--verified-bd)' : statusLimited ? 'var(--amber-bd)' : 'var(--failed-bd)', color: statusOk ? 'var(--verified)' : statusLimited ? 'var(--amber)' : 'var(--failed)' }}>
            {statusText}
          </span>
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 'auto', padding: '14px 18px', borderTop: '1px solid var(--rule)' }}>
        <button
          onClick={onStartOver}
          style={{ ...mono, fontSize: 11, padding: '6px 0', width: '100%', textAlign: 'center', border: '1px solid var(--rule)', borderRadius: 4, background: 'transparent', color: 'var(--ink-2)', cursor: 'pointer', letterSpacing: '0.03em', transition: 'background 0.13s' }}
        >
          ← New problem
        </button>
      </div>
    </div>
  );
};

// ── Main component ────────────────────────────────────────────────────────────
export const PipelineResults: React.FC<{ result: CompletePipelineResponse; onStartOver: () => void }> = ({ result, onStartOver }) => {
  const [activeTab, setActiveTab] = useState<'solution' | 'thinking' | 'code'>('solution');
  const [proofView, setProofView] = useState(false);

  if (!result.data) {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--cream)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ maxWidth: 480, padding: 32, background: 'var(--failed-bg)', border: '1px solid var(--failed-bd)', borderRadius: 8 }}>
          <div style={{ ...mono, fontSize: 13, color: 'var(--failed)', marginBottom: 12 }}>Pipeline failed</div>
          <div style={{ ...serif2, fontSize: 15, color: 'var(--ink-2)', lineHeight: 1.6, marginBottom: 20 }}>{result.message}</div>
          <button onClick={onStartOver} style={{ ...mono, fontSize: 11, padding: '8px 20px', border: '1px solid var(--rule)', borderRadius: 4, background: 'var(--cream)', color: 'var(--ink-2)', cursor: 'pointer' }}>← Start over</button>
        </div>
      </div>
    );
  }

  const { vision, reasoning, verification } = result.data;
  const steps = reasoning.steps ?? [];
  const repairHistory = verification?.repair_history ?? [];
  const errors = verification?.errors ?? [];
  const verificationMetadata = verification?.metadata ?? {};
  const repairAttempts = Math.max(0, repairHistory.length - 1);

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <style>{`
        .midas-tab { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.05em; padding:7px 18px; cursor:pointer; color:var(--ink-3); border-bottom:2px solid transparent; margin-bottom:-1px; transition:all .13s; background:transparent; border-top:none; border-left:none; border-right:none; }
        .midas-tab:hover { color:var(--ink); }
        .midas-tab.active { color:var(--accent); border-bottom-color:var(--accent); }
        .midas-btn-ghost { font-family:'JetBrains Mono',monospace; font-size:11px; padding:6px 14px; border-radius:4px; cursor:pointer; border:1px solid var(--rule); background:transparent; color:var(--ink-2); transition:all .13s; letter-spacing:.02em; }
        .midas-btn-ghost:hover { background:var(--parchment); }
        .midas-btn-on { font-family:'JetBrains Mono',monospace; font-size:11px; padding:6px 14px; border-radius:4px; cursor:pointer; border:1px solid var(--accent); background:var(--accent-lt); color:var(--accent); transition:all .13s; }
        .midas-step-card { border-left:3px solid var(--rule-lt); border-radius:0 5px 5px 0; padding:13px 14px 13px 18px; background:var(--cream); margin-bottom:3px; display:flex; gap:0; transition:background .15s,border-color .15s; }
        .midas-step-card.v { border-left-color:var(--rule-lt); background:var(--cream); }
        .midas-step-card.f { border-left-color:var(--failed); background:var(--failed-bg); }
        .midas-widget { background:var(--parchment); border:1px solid var(--rule); border-radius:6px; overflow:hidden; margin-bottom:14px; }
        .midas-widget-hd { padding:9px 13px; border-bottom:1px solid var(--rule); font-size:9.5px; font-family:'JetBrains Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); display:flex; align-items:center; justify-content:space-between; }
        .midas-widget-bd { padding:12px 13px; }
        .midas-w-row { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; font-size:13px; }
      `}</style>

      {/* Left sidebar */}
      <Sidebar
        problemStatement={reasoning.original_problem}
        steps={steps.length}
        repairAttempts={repairAttempts}
        status={verification?.status ?? 'unknown'}
        confidence={verification?.confidence_score ?? 0}
        onStartOver={onStartOver}
      />

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Page header */}
        <div style={{ padding: '26px 36px 20px', borderBottom: '1px solid var(--rule)', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 20, flexShrink: 0 }}>
          <div>
            <div style={{ ...mono, fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 6 }}>
              Problem statement
            </div>
            <div style={{ fontSize: 20, fontStyle: 'italic', color: 'var(--ink)', lineHeight: 1.45, maxWidth: 580 }}>
              <SmartMathRenderer content={vision.problem_statement} />
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexShrink: 0, marginTop: 4 }}>
            <button className={proofView ? 'midas-btn-ghost' : 'midas-btn-on'} onClick={() => setProofView(false)}>Narrative</button>
            <button className={proofView ? 'midas-btn-on' : 'midas-btn-ghost'} onClick={() => setProofView(true)}>Proof view</button>
          </div>
        </div>

        {/* Body grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 268px', flex: 1, minHeight: 0 }}>

          {/* Solution column */}
          <div style={{ padding: '26px 36px 48px', borderRight: '1px solid var(--rule)', overflowY: 'auto' }} className="thin-scroll">

            {/* Tabs */}
            <div style={{ display: 'flex', borderBottom: '1px solid var(--rule)', marginBottom: 22 }}>
              {(['solution', 'thinking', 'code'] as const).map(t => (
                <button key={t} className={`midas-tab${activeTab === t ? ' active' : ''}`} onClick={() => setActiveTab(t)}>
                  {t === 'code' ? 'SymPy code' : t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {/* Solution tab */}
            {activeTab === 'solution' && (
              <>
                {proofView ? (
                  <>
                    <div style={{ fontSize: 9.5, ...mono, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', paddingBottom: 8, borderBottom: '1px solid var(--rule-lt)', marginBottom: 18, display: 'flex', justifyContent: 'space-between' }}>
                      <span>Proof structure</span>
                      <span style={{ fontStyle: 'italic', fontFamily: "'EB Garamond', serif", letterSpacing: 0, textTransform: 'none' }}>click any step to expand</span>
                    </div>
                    <ProofView problemStatement={reasoning.original_problem} steps={steps} finalAnswer={reasoning.final_answer} />
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 9.5, ...mono, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', paddingBottom: 8, borderBottom: '1px solid var(--rule-lt)', marginBottom: 18, display: 'flex', justifyContent: 'space-between' }}>
                      <span>Solution steps</span>
                      <span style={{ fontStyle: 'italic', fontFamily: "'EB Garamond', serif", letterSpacing: 0, textTransform: 'none' }}>click a failed step for feedback</span>
                    </div>
                    {steps.length > 0
                      ? steps.map(step => <StepCard key={step.step_number} step={step} problemStatement={reasoning.original_problem} />)
                      : <div style={{ ...serif2, fontStyle: 'italic', color: 'var(--ink-3)', fontSize: 15 }}>No structured steps available.</div>
                    }
                  </>
                )}

                {/* Therefore */}
                <div style={{ display: 'flex', gap: 16, marginTop: 22, paddingTop: 16, borderTop: '1.5px solid var(--rule)', alignItems: 'baseline' }}>
                  <div style={{ ...mono, fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-3)', width: 68, flexShrink: 0 }}>Therefore</div>
                  <div style={{ fontSize: 22, fontWeight: 600, fontStyle: 'italic', color: 'var(--accent)' }}>
                    <SmartMathRenderer content={reasoning.final_answer} />
                  </div>
                </div>

                <RepairHistory repairHistory={repairHistory} />
              </>
            )}

            {/* Thinking tab */}
            {activeTab === 'thinking' && (
              <>
                <div style={{ fontSize: 9.5, ...mono, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', paddingBottom: 8, borderBottom: '1px solid var(--rule-lt)', marginBottom: 18 }}>
                  Internal scratchpad
                </div>
                <div style={{ background: 'var(--parchment)', border: '1px solid var(--rule)', borderRadius: 6, padding: '14px 16px', ...mono, fontSize: 11.5, lineHeight: 1.75, color: 'var(--ink-2)', maxHeight: 360, overflow: 'auto' }}>
                  {reasoning.think_reasoning || <span style={{ color: 'var(--ink-3)', fontStyle: 'italic' }}>No thinking trace available.</span>}
                </div>
              </>
            )}

            {/* Code tab */}
            {activeTab === 'code' && (
              <>
                <div style={{ fontSize: 9.5, ...mono, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', paddingBottom: 8, borderBottom: '1px solid var(--rule-lt)', marginBottom: 18 }}>
                  Generated SymPy verification code
                </div>
                <div style={{ background: '#1e1e1e', borderRadius: 6, padding: 16, overflow: 'auto', maxHeight: 400 }}>
                  <pre style={{ ...mono, fontSize: 11.5, lineHeight: 1.75, color: '#a8d8a0', margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                    {verification?.generated_code || '# No code generated'}
                  </pre>
                </div>
                <div style={{ ...mono, fontSize: 11, color: 'var(--ink-3)', marginTop: 8 }}>
                  codegen/baseline_codegen@v6
                </div>
              </>
            )}
          </div>

          {/* Right sidebar: verification */}
          <div style={{ padding: '26px 22px 40px', overflowY: 'auto' }} className="thin-scroll">

            <div style={{ fontSize: 9.5, ...mono, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 12 }}>
              Verification
            </div>

            {/* Summary widget */}
            {verification && (
              <div className="midas-widget">
                <div className="midas-widget-hd">
                  result
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 8px', borderRadius: 3, fontSize: 10.5, border: '1px solid', ...mono, background: verification.status === 'verified' ? 'var(--verified-bg)' : verification.status === 'unsupported' || verification.status === 'needs_visual_context' ? 'var(--amber-bg)' : 'var(--failed-bg)', borderColor: verification.status === 'verified' ? 'var(--verified-bd)' : verification.status === 'unsupported' || verification.status === 'needs_visual_context' ? 'var(--amber-bd)' : 'var(--failed-bd)', color: verification.status === 'verified' ? 'var(--verified)' : verification.status === 'unsupported' || verification.status === 'needs_visual_context' ? 'var(--amber)' : 'var(--failed)' }}>
                    {verification.status}
                  </span>
                </div>
                <div className="midas-widget-bd">
                  {verification.status === 'unsupported' && (
                    <div style={{ ...serif2, color: 'var(--ink-2)', fontStyle: 'italic', fontSize: 13, lineHeight: 1.5, marginBottom: 10 }}>
                      Solved, but symbolic verification is not available for this problem type.
                    </div>
                  )}
                  {verification.status === 'needs_visual_context' && (
                    <div style={{ ...serif2, color: 'var(--ink-2)', fontStyle: 'italic', fontSize: 13, lineHeight: 1.5, marginBottom: 10 }}>
                      This problem appears to depend on a diagram, graph, or table that was not available to the reasoning model.
                    </div>
                  )}
                  <div className="midas-w-row"><span style={{ ...serif2, color: 'var(--ink-3)' }}>Steps verified</span><span style={{ ...mono, fontSize: 11.5, color: steps.filter(s => s.verification_status === true).length === steps.length ? 'var(--verified)' : 'var(--ink)' }}>{steps.filter(s => s.verification_status === true).length} / {steps.length}</span></div>
                  <div className="midas-w-row"><span style={{ ...serif2, color: 'var(--ink-3)' }}>Steps failed</span><span style={{ ...mono, fontSize: 11.5, color: steps.filter(s => s.verification_status === false).length > 0 ? 'var(--failed)' : 'var(--ink)' }}>{steps.filter(s => s.verification_status === false).length}</span></div>
                  {verificationMetadata.visual_context_required === true && <div className="midas-w-row"><span style={{ ...serif2, color: 'var(--ink-3)' }}>Visual context</span><span style={{ ...mono, fontSize: 11.5 }}>{verificationMetadata.visual_context_attached ? 'attached' : 'needed'}</span></div>}
                  <div className="midas-w-row"><span style={{ ...serif2, color: 'var(--ink-3)' }}>Confidence</span><span style={{ ...mono, fontSize: 11.5 }}>{(verification.confidence_score * 100).toFixed(1)}%</span></div>
                  {/* Step dots */}
                  <div style={{ display: 'flex', gap: 5, marginTop: 10, flexWrap: 'wrap' }}>
                    {steps.map(s => (
                      <div key={s.step_number} title={`Step ${s.step_number}`} style={{ width: 11, height: 11, borderRadius: '50%', background: s.verification_status === true ? 'var(--verified)' : s.verification_status === false ? 'var(--failed)' : 'var(--rule)' }} />
                    ))}
                  </div>
                  <DetailsBox>
                    <div>Repair attempts: {repairAttempts}</div>
                    {verificationMetadata.unsupported_reason && <div>Boundary: {String(verificationMetadata.unsupported_reason).replaceAll('_', ' ')}</div>}
                    {verificationMetadata.visual_context_required !== undefined && (
                      <div>Visual context: {verificationMetadata.visual_context_attached ? 'attached' : verificationMetadata.visual_context_required ? 'needed' : 'not needed'}</div>
                    )}
                    <div>Processing time: {verification.processing_time?.toFixed ? verification.processing_time.toFixed(2) : verification.processing_time}s</div>
                  </DetailsBox>
                </div>
              </div>
            )}

            {/* Errors */}
            {errors.length > 0 && errors.map((err, i) => {
              const meta = ERR_META[err.error_type] ?? { label: err.error_type, description: '' };
              const isFault = err.error_type === 'ANSWER_MISMATCH' || err.error_type === 'ASSERTION_FAILED';
              return (
                <div key={i} style={{ background: isFault ? 'var(--failed-bg)' : 'var(--amber-bg)', border: `1px solid ${isFault ? 'var(--failed-bd)' : 'var(--amber-bd)'}`, borderRadius: 6, overflow: 'hidden', marginBottom: 14 }}>
                  <div style={{ padding: '8px 13px', borderBottom: `1px solid ${isFault ? 'var(--failed-bd)' : 'var(--amber-bd)'}`, ...mono, fontSize: 9.5, letterSpacing: '0.12em', textTransform: 'uppercase', color: isFault ? 'var(--failed)' : 'var(--amber)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    {meta.label}
                    <span style={{ fontSize: 9.5, background: isFault ? 'var(--failed)' : 'var(--amber)', color: 'white', padding: '1px 6px', borderRadius: 2, ...mono }}>
                      {err.error_type.toLowerCase().replace('_', ' ')}
                    </span>
                  </div>
                  <div style={{ padding: '10px 13px', fontSize: 13, color: 'var(--ink-2)', ...serif2, lineHeight: 1.55 }}>
                    {meta.description || 'Verification could not complete cleanly.'}
                    <DetailsBox>
                      <div>{err.message}</div>
                      {err.suggested_fix && <div style={{ marginTop: 8 }}>Suggestion: {err.suggested_fix}</div>}
                      {err.traceback && <div style={{ marginTop: 8 }}>{err.traceback}</div>}
                    </DetailsBox>
                  </div>
                </div>
              );
            })}

            {/* Trajectory metadata */}
            {repairAttempts >= 0 && (
              <DetailsBox title="Run details">
                <div>attempt_count: {repairAttempts + 1}</div>
                <div>steps: {steps.length}</div>
                <div>prompt_version: solve@v2</div>
              </DetailsBox>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
