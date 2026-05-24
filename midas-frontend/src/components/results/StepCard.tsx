import React, { useState } from 'react';
import { ReasoningStepResponse, FeedbackRequest } from '../../types/api';
import { SmartMathRenderer } from '../ui/SmartMathRenderer';
import { SimpleAPIService } from '../../services/SimpleAPIService';

interface StepCardProps {
  step: ReasoningStepResponse;
  problemStatement: string;
}

export const StepCard: React.FC<StepCardProps> = ({ step, problemStatement }) => {
  const [feedback, setFeedback] = useState<string | null>(step.feedback || null);
  const [loadingFeedback, setLoadingFeedback] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const statusBorder =
    step.verification_status === false ? 'var(--failed)' : 'var(--rule-lt)';

  const handleGetFeedback = async () => {
    setLoadingFeedback(true);
    try {
      const req: FeedbackRequest = {
        problem_statement: problemStatement,
        step_number: step.step_number,
        claim: step.claim,
        latex_expression: step.latex_expression,
        justification: step.justification,
        verification_note: step.verification_note,
      };
      const resp = await SimpleAPIService.getFeedback(req);
      if (resp.success && resp.data) {
        setFeedback(resp.data.feedback);
      }
    } finally {
      setLoadingFeedback(false);
    }
  };

  return (
    <div
      style={{
        borderLeft: `3px solid ${statusBorder}`,
        padding: '13px 14px 13px 18px',
        marginBottom: 6,
        borderRadius: '0 5px 5px 0',
        background: 'var(--cream)',
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono font-bold text-sm" style={{ color: step.verification_status === false ? 'var(--failed)' : 'var(--ink-3)' }}>
          Step {step.step_number}
        </span>
        {step.verification_status === true && (
          <span
            className="text-xs"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              color: 'var(--verified)',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10.5,
            }}
          >
            <span style={{ width: 15, height: 15, borderRadius: '50%', background: 'var(--verified)', color: 'white', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, lineHeight: 1 }}>✓</span>
            Verified
          </span>
        )}
        {step.verification_status === false && (
          <span className="text-xs" style={{ color: 'var(--failed)', fontFamily: "'JetBrains Mono', monospace", fontSize: 10.5 }}>Verification failed</span>
        )}
      </div>

      <div className="text-sm text-gray-900 mb-1">
        <SmartMathRenderer content={step.claim} />
      </div>

      {step.latex_expression && (
        <div className="text-sm my-1 py-1">
          <SmartMathRenderer content={step.latex_expression} mathOnly />
        </div>
      )}

      <div className="text-xs text-gray-500 italic">
        <SmartMathRenderer content={step.justification} />
      </div>

      {step.verification_status === false && step.verification_note && (
        <details
          open={detailsOpen}
          onToggle={(event) => setDetailsOpen(event.currentTarget.open)}
          className="mt-2"
        >
          <summary
            style={{
              cursor: 'pointer',
              color: 'var(--failed)',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              listStyle: 'none',
            }}
          >
            Details
          </summary>
          <div className="mt-2 font-mono text-xs px-2 py-1 rounded" style={{ color: 'var(--failed)', background: 'var(--failed-bg)', border: '1px solid var(--failed-bd)' }}>
            {step.verification_note}
          </div>
        </details>
      )}

      {step.verification_status === false && feedback && (
        <div className="mt-2 text-sm text-red-900 bg-red-50 border border-red-200 rounded px-3 py-2">
          <span className="font-semibold text-xs uppercase tracking-wide text-red-700 block mb-1">Feedback</span>
          {feedback}
        </div>
      )}

      {step.verification_status === false && !feedback && (
        <button
          onClick={handleGetFeedback}
          disabled={loadingFeedback}
          className="mt-2 text-xs text-red-600 underline hover:text-red-800 disabled:opacity-50"
        >
          {loadingFeedback ? 'Loading...' : 'Get explanation →'}
        </button>
      )}
    </div>
  );
};
