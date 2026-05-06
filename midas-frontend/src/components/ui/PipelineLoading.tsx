import React, { useEffect, useState } from 'react';
import { ProcessingStage } from '../../types/api';

interface PipelineLoadingProps {
  stage: ProcessingStage;
  message?: string;
}

type SidebarState = 'done' | 'active' | 'wait';

interface StageConfig {
  eyebrow: string;
  name: string;
  desc: string;
  nodes: SidebarState[];   // [vision, reasoning, verification, results]
  substeps: Array<{ label: string; status: 'done' | 'active' | 'wait' }>;
}

const STAGE_CONFIG: Record<string, StageConfig> = {
  vision: {
    eyebrow: 'Stage 1 of 3',
    name: 'Vision',
    desc: 'Processing the document with OCR and layout analysis to extract the mathematical content.',
    nodes: ['active', 'wait', 'wait', 'wait'],
    substeps: [
      { label: 'Running Marker OCR', status: 'active' },
      { label: 'Extracting text blocks', status: 'wait' },
      { label: 'Grouping problems semantically', status: 'wait' },
      { label: 'Linking blocks to problems', status: 'wait' },
    ],
  },
  reasoning: {
    eyebrow: 'Stage 2 of 3',
    name: 'Reasoning',
    desc: 'The model is working through the solution step by step, constructing a structured proof.',
    nodes: ['done', 'active', 'wait', 'wait'],
    substeps: [
      { label: 'Preparing prompt (solve@v2)', status: 'done' },
      { label: 'Model inference', status: 'active' },
      { label: 'Parsing structured steps', status: 'wait' },
      { label: 'Extracting LaTeX expressions', status: 'wait' },
    ],
  },
  verification: {
    eyebrow: 'Stage 3 of 3',
    name: 'Verification',
    desc: 'Generating and executing SymPy code to check each step symbolically.',
    nodes: ['done', 'done', 'active', 'wait'],
    substeps: [
      { label: 'Generating SymPy code (v6)', status: 'done' },
      { label: 'Sandboxed execution', status: 'active' },
      { label: 'Annotating reasoning steps', status: 'wait' },
      { label: 'Generating student feedback', status: 'wait' },
    ],
  },
};

const stageToConfig = (stage: ProcessingStage): StageConfig => {
  if (stage === 'validating' || stage === 'processing' || stage === 'analyzing' || stage === 'uploading')
    return STAGE_CONFIG.vision;
  if (stage === 'writing_code' || stage === 'solving')
    return STAGE_CONFIG.verification;
  return STAGE_CONFIG.reasoning; // thinking, etc.
};

const NODE_LABELS = ['Vision', 'Reasoning', 'Verify', 'Results'];

const sb: React.CSSProperties = {
  width: 220, flexShrink: 0,
  background: 'var(--parchment)',
  borderRight: '1px solid var(--rule)',
  display: 'flex', flexDirection: 'column',
  overflow: 'hidden',
};

export const PipelineLoading: React.FC<PipelineLoadingProps> = ({ stage }) => {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed(s => s + 0.1), 100);
    return () => clearInterval(t);
  }, []);

  const cfg = stageToConfig(stage);

  const dotStyle = (s: SidebarState): React.CSSProperties => ({
    width: 20, height: 20, borderRadius: '50%',
    border: `1.5px solid ${s === 'done' ? 'var(--verified-bd)' : s === 'active' ? 'var(--accent)' : 'var(--rule)'}`,
    background: s === 'done' ? 'var(--verified-bg)' : s === 'active' ? 'var(--accent-lt)' : 'var(--cream)',
    color: s === 'done' ? 'var(--verified)' : s === 'active' ? 'var(--accent)' : 'var(--rule)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 9, fontFamily: "'JetBrains Mono', monospace",
    flexShrink: 0,
    animation: s === 'active' ? 'midas-pulse 1.4s ease-in-out infinite' : 'none',
  });

  const sbNames = ['Vision', 'Reasoning', 'Verification', 'Results'];

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <style>{`
        @keyframes midas-pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
        @keyframes midas-shimmer { 0%{left:-100%} 100%{left:100%} }
        @keyframes midas-node-pulse { 0%,100%{box-shadow:0 0 0 6px rgba(30,58,95,.08)} 50%{box-shadow:0 0 0 10px rgba(30,58,95,.04)} }
        @keyframes midas-blink { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes midas-indeterminate { 0%{transform:translateX(-100%) scaleX(.4)} 50%{transform:translateX(100%) scaleX(.8)} 100%{transform:translateX(300%) scaleX(.4)} }
      `}</style>

      {/* Sidebar */}
      <div style={sb}>
        <div style={{ padding: '20px 18px 16px', borderBottom: '1px solid var(--rule)' }}>
          <div style={{ fontSize: 19, fontWeight: 600, letterSpacing: '0.06em' }}>MIDAS</div>
          <div style={{ fontSize: 10.5, fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-3)', marginTop: 2 }}>
            phi4-mini-reasoning
          </div>
        </div>
        <div style={{ padding: '14px 18px 6px' }}>
          <div style={{ fontSize: 9.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 8 }}>
            Pipeline
          </div>
          {sbNames.map((name, i) => {
            const s = cfg.nodes[i];
            return (
              <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 8px', borderRadius: 4, marginBottom: 2, background: s === 'active' ? 'var(--accent-lt)' : 'transparent' }}>
                <div style={dotStyle(s)}>
                  {s === 'done' ? '✓' : s === 'active' ? '◌' : i + 1}
                </div>
                <div style={{ fontSize: 13.5, fontFamily: "'Crimson Pro', serif", color: s === 'active' ? 'var(--accent)' : s === 'wait' ? 'var(--rule)' : 'var(--ink-2)', fontWeight: s === 'active' ? 600 : 400 }}>
                  {name}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40, background: 'var(--cream)', position: 'relative', overflow: 'hidden' }}>
        <div style={{ maxWidth: 520, width: '100%', textAlign: 'center' }}>

          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 10 }}>
            {cfg.eyebrow}
          </div>
          <div style={{ fontSize: 28, fontWeight: 500, color: 'var(--ink)', marginBottom: 8, lineHeight: 1.2 }}>
            {cfg.name}
          </div>
          <div style={{ fontSize: 15, fontStyle: 'italic', color: 'var(--ink-3)', fontFamily: "'Crimson Pro', serif", marginBottom: 40, lineHeight: 1.5 }}>
            {cfg.desc}
          </div>

          {/* Pipeline track */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 40 }}>
            {NODE_LABELS.map((label, i) => {
              const s = cfg.nodes[i];
              return (
                <React.Fragment key={label}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      width: 36, height: 36, borderRadius: '50%',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
                      background: s === 'done' ? 'var(--verified-bg)' : s === 'active' ? 'var(--accent-lt)' : 'var(--parchment)',
                      border: `2px solid ${s === 'done' ? 'var(--verified)' : s === 'active' ? 'var(--accent)' : 'var(--rule)'}`,
                      color: s === 'done' ? 'var(--verified)' : s === 'active' ? 'var(--accent)' : 'var(--rule)',
                      animation: s === 'active' ? 'midas-node-pulse 2s ease-in-out infinite' : 'none',
                    }}>
                      {s === 'done' ? '✓' : s === 'active' ? '◌' : i + 1}
                    </div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9.5, letterSpacing: '0.06em', textTransform: 'uppercase', color: s === 'done' ? 'var(--verified)' : s === 'active' ? 'var(--accent)' : 'var(--rule)', width: 60, textAlign: 'center', fontWeight: s === 'active' ? 500 : 400 }}>
                      {label}
                    </div>
                  </div>
                  {i < NODE_LABELS.length - 1 && (
                    <div style={{ width: 44, height: 2, background: cfg.nodes[i] === 'done' ? 'var(--verified)' : 'var(--rule)', marginBottom: 24, position: 'relative', overflow: 'hidden' }}>
                      {cfg.nodes[i] === 'active' && (
                        <div style={{ position: 'absolute', top: 0, left: '-100%', width: '100%', height: '100%', background: 'linear-gradient(90deg, transparent, var(--accent), transparent)', animation: 'midas-shimmer 1.6s ease-in-out infinite' }} />
                      )}
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Substep log */}
          <div style={{ background: 'var(--parchment)', border: '1px solid var(--rule)', borderRadius: 7, overflow: 'hidden', textAlign: 'left', marginBottom: 28 }}>
            <div style={{ padding: '9px 14px', borderBottom: '1px solid var(--rule)', fontFamily: "'JetBrains Mono', monospace", fontSize: 9.5, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-3)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--accent)', animation: 'midas-blink 1.2s ease-in-out infinite' }} />
              {cfg.name.toLowerCase()} — in progress
            </div>
            <div style={{ padding: '10px 0' }}>
              {cfg.substeps.map((sub, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 14px', fontFamily: "'JetBrains Mono', monospace", fontSize: 11.5, background: sub.status === 'active' ? 'var(--accent-lt)' : 'transparent', color: sub.status === 'done' ? 'var(--ink-2)' : sub.status === 'active' ? 'var(--ink)' : 'var(--rule)' }}>
                  <div style={{ width: 14, height: 14, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 8, flexShrink: 0, background: sub.status === 'done' ? 'var(--verified)' : sub.status === 'active' ? 'var(--accent-lt)' : 'var(--rule-lt)', border: sub.status === 'active' ? '1.5px solid var(--accent)' : 'none', color: sub.status === 'done' ? 'white' : sub.status === 'active' ? 'var(--accent)' : 'var(--rule)', animation: sub.status === 'active' ? 'midas-pulse 1.4s ease-in-out infinite' : 'none' }}>
                    {sub.status === 'done' ? '✓' : sub.status === 'active' ? '·' : ''}
                  </div>
                  {sub.label}
                </div>
              ))}
            </div>
          </div>

          {/* Elapsed */}
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: 'var(--ink-3)', display: 'flex', alignItems: 'center', gap: 10, justifyContent: 'center' }}>
            <span>{elapsed.toFixed(1)}s</span>
            <div style={{ width: 120, height: 2, background: 'var(--rule)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', background: 'var(--accent)', borderRadius: 2, animation: 'midas-indeterminate 2s ease-in-out infinite' }} />
            </div>
            <span>processing…</span>
          </div>

        </div>
      </div>
    </div>
  );
};
