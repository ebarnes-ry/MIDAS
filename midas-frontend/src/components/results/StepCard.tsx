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

  const statusBorder =
    step.verification_status === true  ? 'border-l-green-500 bg-green-50' :
    step.verification_status === false ? 'border-l-red-500 bg-red-50' :
                                          'border-l-gray-200 bg-white';

  const statusIcon =
    step.verification_status === true  ? '✓' :
    step.verification_status === false ? '✗' : '·';

  const statusColor =
    step.verification_status === true  ? 'text-green-600' :
    step.verification_status === false ? 'text-red-600' : 'text-gray-400';

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
    <div className={`border-l-4 pl-4 py-3 mb-2 rounded-r transition-colors ${statusBorder}`}>
      <div className="flex items-baseline gap-2 mb-1">
        <span className={`font-mono font-bold text-sm ${statusColor}`}>
          {statusIcon} Step {step.step_number}
        </span>
        {step.verification_status === true && (
          <span className="text-xs text-green-600">Verified by SymPy</span>
        )}
        {step.verification_status === false && (
          <span className="text-xs text-red-600">Verification failed</span>
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
        <div className="mt-2 font-mono text-xs text-red-700 bg-red-50 border border-red-100 px-2 py-1 rounded">
          SymPy: {step.verification_note}
        </div>
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
