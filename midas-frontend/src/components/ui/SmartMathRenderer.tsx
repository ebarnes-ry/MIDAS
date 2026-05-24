import React from 'react';
import { MathJax } from 'better-react-mathjax';
import { MathErrorBoundary } from '../vision/VisionErrorBoundary';
import { looksLikeHTML, normalizeMathForRendering } from './mathml';

interface SmartMathRendererProps {
  content: string;
  mathOnly?: boolean;
}

const sanitizeHTML = (html: string): string =>
  html.trim()
    .replaceAll(/<math\b[^>]*>/g, '$')
    .replaceAll(/<\/math>/g, '$');

/**
 * Encodes only the characters that would create unintended HTML structure
 * (&, <, >) while leaving $ and \ intact so MathJax can find them.
 */
const encodeForDOM = (text: string): string =>
  text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

/**
 * Reliably renders LaTeX mixed with prose.
 *
 * THE CRITICAL DETAIL: better-react-mathjax's MathJax.typesetPromise fires via
 * a MutationObserver that watches for real DOM mutations. React-managed text
 * nodes (JSX string children) do not always trigger that observer, so MathJax
 * silently skips them. Using dangerouslySetInnerHTML writes directly into the
 * DOM and consistently triggers typesetting for both HTML and plain-text paths.
 *
 * For HTML content (Marker OCR output): sanitise <math> tags first.
 * For plain text: HTML-encode <, >, & so no tags are injected, but leave
 * $ and \ intact for MathJax to find and typeset.
 */
export const SmartMathRenderer: React.FC<SmartMathRendererProps> = ({ content, mathOnly = false }) => {
  if (!content) return null;

  const normalizedContent = normalizeMathForRendering(content, mathOnly);

  const html = looksLikeHTML(normalizedContent)
    ? sanitizeHTML(normalizedContent)
    : encodeForDOM(normalizedContent);

  return (
    <MathErrorBoundary>
      <MathJax dynamic>
        <span
          style={{ whiteSpace: 'pre-wrap' }}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </MathJax>
    </MathErrorBoundary>
  );
};
