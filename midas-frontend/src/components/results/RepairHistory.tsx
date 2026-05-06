import React, { useState } from 'react';
import { RepairAttempt, ReasoningStepResponse } from '../../types/api';

interface RepairHistoryProps {
  repairHistory: RepairAttempt[];
}

const diffAttempts = (
  prev: ReasoningStepResponse[],
  curr: ReasoningStepResponse[]
): Array<{ step_number: number; prev_claim: string; curr_claim: string; changed: boolean }> => {
  const prevMap = new Map(prev.map(s => [s.step_number, s]));
  const currMap = new Map(curr.map(s => [s.step_number, s]));
  const allNums = new Set([...prevMap.keys(), ...currMap.keys()]);

  return Array.from(allNums).sort().map(n => {
    const p = prevMap.get(n);
    const c = currMap.get(n);
    return {
      step_number: n,
      prev_claim: p?.claim || '(step removed)',
      curr_claim: c?.claim || '(step added)',
      changed: p?.claim !== c?.claim,
    };
  });
};

/**
 * Shows a diff between repair attempts — what changed between attempt N and N+1.
 * Only renders when there are 2+ attempts (initial + at least one repair).
 */
export const RepairHistory: React.FC<RepairHistoryProps> = ({ repairHistory }) => {
  const [expanded, setExpanded] = useState(false);

  if (!repairHistory || repairHistory.length < 2) return null;

  return (
    <div className="mt-4 border border-amber-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-amber-50 hover:bg-amber-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-amber-600">⟳</span>
          <span className="font-medium text-sm text-amber-900">
            Repair history — {repairHistory.length - 1} repair attempt{repairHistory.length - 1 !== 1 ? 's' : ''}
          </span>
          <span className="text-xs text-amber-600 bg-amber-100 px-2 py-0.5 rounded-full">
            {repairHistory[repairHistory.length - 1].verification_status}
          </span>
        </div>
        <span className="text-amber-600 text-xs">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="p-4 bg-white space-y-4">
          <p className="text-xs text-gray-500">
            The reasoning model made {repairHistory.length - 1} repair attempt{repairHistory.length - 1 !== 1 ? 's' : ''}.
            Each attempt fed the previous verification errors back as input.
            Steps that changed between attempts are highlighted.
          </p>

          {repairHistory.slice(1).map((attempt, idx) => {
            const prev = repairHistory[idx];
            const diffs = diffAttempts(
              prev.reasoning_steps || [],
              attempt.reasoning_steps || []
            );
            const changedCount = diffs.filter(d => d.changed).length;

            return (
              <div key={attempt.attempt_number} className="border border-gray-100 rounded p-3">
                <div className="flex items-center gap-2 mb-3">
                  <span className="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">
                    Attempt {attempt.attempt_number}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    attempt.verification_status === 'verified'
                      ? 'bg-green-100 text-green-700'
                      : 'bg-red-100 text-red-700'
                  }`}>
                    {attempt.verification_status}
                  </span>
                  <span className="text-xs text-gray-500">
                    {changedCount} step{changedCount !== 1 ? 's' : ''} changed
                  </span>
                </div>

                {diffs.length > 0 ? (
                  <div className="space-y-2">
                    {diffs.map(diff => (
                      <div
                        key={diff.step_number}
                        className={`text-xs rounded p-2 ${
                          diff.changed ? 'bg-amber-50 border border-amber-100' : 'bg-gray-50'
                        }`}
                      >
                        <span className="font-mono text-gray-500 mr-2">Step {diff.step_number}</span>
                        {diff.changed ? (
                          <div className="mt-1 space-y-1">
                            <div className="text-red-600 line-through opacity-60">{diff.prev_claim}</div>
                            <div className="text-green-700">{diff.curr_claim}</div>
                          </div>
                        ) : (
                          <span className="text-gray-600">{diff.curr_claim}</span>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-gray-400 italic">No step data available for this attempt.</p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
