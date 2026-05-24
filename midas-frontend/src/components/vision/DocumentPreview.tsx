import React from 'react';
import { Button } from '../ui/Button';

interface DocumentPreviewProps {
  imageBase64: string;
  fileName: string;
  onProcess: () => void;
  onCancel: () => void;
  isProcessing?: boolean;
}

export const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  imageBase64,
  fileName,
  onProcess,
  onCancel,
  isProcessing = false
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl w-full">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          {/* Header */}
          <div className="p-4 border-b border-gray-200 bg-gray-50">
            <h2 className="text-lg font-semibold text-gray-900">Document Preview</h2>
            <p className="text-sm text-gray-600 mt-1">Ready to process: {fileName}</p>
            <p className="text-sm text-gray-500 mt-2 italic">
              Fresh uploads run OCR, layout analysis, and grouping live. This can take several minutes.
            </p>
          </div>

          {/* Image Preview */}
          <div className="p-4">
            <div className="border border-gray-200 rounded-lg overflow-hidden bg-gray-50">
              <img
                src={`data:image/png;base64,${imageBase64}`}
                alt="Document preview"
                className="w-full h-auto max-h-96 object-contain"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="flex justify-between items-center">
              <Button
                variant="secondary"
                onClick={onCancel}
                disabled={isProcessing}
              >
                Cancel
              </Button>
              <Button
                onClick={onProcess}
                disabled={isProcessing}
                loading={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Process Document'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
