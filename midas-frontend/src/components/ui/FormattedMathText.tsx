import React, { useMemo } from 'react';
import { MathJax } from 'better-react-mathjax';
import { MathErrorBoundary } from '../vision/VisionErrorBoundary';
import { looksLikeHTML, normalizeMathForRendering } from './mathml';

interface FormattedMathTextProps {
  content: string;
  fontSize?: number;
  lineHeight?: number;
}

const encodeForDOM = (text: string): string =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const sanitizeHTML = (html: string): string =>
  html.replaceAll(/<math\b[^>]*>/g, '$').replaceAll(/<\/math>/g, '$');

/**
 * Renders OCR-extracted problem text with proper spacing and indentation,
 * with all maths typeset by MathJax in a SINGLE pass.
 *
 * Each \n becomes its own block element. Blank lines become visible gaps.
 * Leading whitespace is converted to proportional left-padding so that
 * indented solution steps retain their visual structure. Display-math-only
 * lines ($$...$$) are centred.
 *
 * THE KEY: we build a single HTML string from all lines and inject it via one
 * dangerouslySetInnerHTML inside one <MathJax dynamic> wrapper. This means
 * MathJax runs typesetPromise() exactly once for the whole block, avoiding
 * the race condition that occurs when each line has its own MathJax instance.
 */
export const FormattedMathText: React.FC<FormattedMathTextProps> = ({
  content,
  fontSize = 15,
  lineHeight = 1.7,
}) => {
  const html = useMemo(() => {
    if (!content) return '';

    const lines = content.split('\n');

    return lines.map(line => {
      // Blank line → paragraph gap
      if (!line.trim()) {
        return `<div style="height:${Math.round(fontSize * 0.6)}px"></div>`;
      }

      const trimmed = line.trim();

      // Leading whitespace → indent
      const leadingChars = line.match(/^(\s+)/)?.[1] ?? '';
      const paddingLeft = Math.min(
        leadingChars.split('').reduce((n, ch) => n + (ch === '\t' ? 28 : 10), 0),
        80,
      );

      // Display-math-only line
      const isDisplayMath =
        trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4;

      const normalized = normalizeMathForRendering(trimmed);
      const encoded = looksLikeHTML(normalized)
        ? sanitizeHTML(normalized)
        : encodeForDOM(normalized);

      const style = [
        `padding-left:${isDisplayMath ? 0 : paddingLeft}px`,
        `margin-bottom:${Math.round(fontSize * 0.25)}px`,
        `text-align:${isDisplayMath ? 'center' : 'left'}`,
        'line-height:' + lineHeight,
      ].join(';');

      return `<div style="${style}">${encoded}</div>`;
    }).join('');
  }, [content, fontSize, lineHeight]);

  if (!html) return null;

  return (
    <MathErrorBoundary>
      <MathJax dynamic>
        <div
          style={{ fontSize }}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </MathJax>
    </MathErrorBoundary>
  );
};
