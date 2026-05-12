import React from 'react';
import { ReasoningStepResponse } from '../../types/api';
import { SmartMathRenderer } from '../ui/SmartMathRenderer';

interface ProofViewProps {
  problemStatement: string;
  steps: ReasoningStepResponse[];
  finalAnswer: string;
}

/**
 * Renders the solution in a formal proof style:
 *   Given → numbered steps (claim + justification) → Therefore
 *
 * Demonstrates proof-structured math education — the core argument of the dissertation.
 */
export const ProofView: React.FC<ProofViewProps> = ({ problemStatement, steps, finalAnswer }) => {
  return (
    <div className="font-serif text-gray-800 space-y-3 p-4 border border-gray-200 rounded-lg bg-gray-50">
      <div className="pb-2 border-b border-gray-200">
        <span className="text-xs font-sans font-semibold uppercase tracking-wide text-gray-500 mr-2">
          Given
        </span>
        <span className="text-sm"><SmartMathRenderer content={problemStatement} /></span>
      </div>

      <div className="space-y-2">
        {steps.map(step => (
          <div key={step.step_number} className="flex gap-3 items-start">
            <span className="font-mono text-xs text-gray-400 mt-1 w-6 flex-shrink-0">
              {step.step_number}.
            </span>
            <div className="flex-1">
              <div className="flex items-start gap-2 flex-wrap">
                <span className="text-sm"><SmartMathRenderer content={step.claim} /></span>
                {step.verification_status === true && (
                  <span className="text-green-500 text-xs mt-0.5">✓</span>
                )}
                {step.verification_status === false && (
                  <span className="text-red-500 text-xs mt-0.5">✗</span>
                )}
              </div>
              {step.latex_expression && (
                <div className="mt-0.5 text-sm text-gray-700">
                  <SmartMathRenderer content={step.latex_expression} mathOnly />
                </div>
              )}
              <div className="text-xs text-gray-400 mt-0.5 font-sans italic">
                by <SmartMathRenderer content={step.justification} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="pt-2 border-t border-gray-200">
        <span className="text-xs font-sans font-semibold uppercase tracking-wide text-gray-500 mr-2">
          Therefore
        </span>
        <span className="text-sm font-semibold"><SmartMathRenderer content={finalAnswer} /></span>
      </div>
    </div>
  );
};
