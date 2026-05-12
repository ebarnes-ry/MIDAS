// Heuristics shared by SmartMathRenderer, FormattedMathText, and CanvasView

export const looksLikeHTML = (s: string) => /<[^>]+>/.test(s);

/**
 * Model responses sometimes contain JSON-escaped TeX after decoding, e.g.
 * "\\(\cos x\\)" instead of "\(\cos x\)". MathJax treats the doubled
 * backslash as a literal character, so delimiters and commands render raw.
 *
 * Collapse doubled backslashes only when they introduce a TeX command or
 * delimiter. This preserves TeX line breaks like "\\ " because those are not
 * followed by a command/delimiter character.
 */
export const normalizeEscapedTeX = (text: string): string =>
  (text ?? '').replace(/\\\\(?=(?:[()[\]{}])|[A-Za-z])/g, '\\');

export const hasTeXDelimiters = (s: string) =>
  /^\s*(\$\$[\s\S]*\$\$|\$[\s\S]*\$|\\\[([\s\S]*)\\\]|\\\(([\s\S]*)\\\))\s*$/.test(normalizeEscapedTeX(s));

export const ensureDelimiters = (s: string) => {
  const t = normalizeEscapedTeX(s).trim();
  if (!t) return t;
  if (hasTeXDelimiters(t)) return t;
  const display = t.includes('\n') || /\\begin|\\frac|=/.test(t);
  return display ? `$$ ${t} $$` : `\\(${t}\\)`;
};

/**
 * Converts $...$ and $$...$$ delimiters to \(...\) and \[...\].
 *
 * MathJax has a known ambiguity with the $ delimiter: when a $ is immediately
 * followed by a digit or space (e.g. "=$1 - \cos…" or "costs $5"), the parser
 * conservatively treats it as a currency sign rather than opening inline math.
 * Any $ left unmatched after that point causes the rest of the paragraph to
 * render as raw LaTeX.
 *
 * The \( \) form is completely unambiguous — MathJax always treats it as math.
 * This function rewrites the text before it reaches MathJax so there are no
 * $ signs left for the parser to misinterpret.
 *
 * Algorithm:
 *   1. Replace $$...$$ with \[...\]  (display math, non-greedy)
 *   2. Walk the remaining string character-by-character. Each $ toggles
 *      between "in math" and "in prose", emitting \( or \) as appropriate.
 *      An unclosed math region at end-of-string is auto-closed with \).
 */
export const normalizeDollarDelimiters = (text: string): string => {
  text = normalizeEscapedTeX(text);

  // Pass 1: display math $$...$$ → \[...\]
  let s = text.replace(/\$\$([\s\S]*?)\$\$/g, (_m, inner) => `\\[${inner}\\]`);

  // Pass 2: inline math $...$ → \(...\)
  let out = '';
  let inMath = false;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '$' && s[i - 1] !== '\\') {
      out += inMath ? '\\)' : '\\(';
      inMath = !inMath;
    } else {
      out += s[i];
    }
  }
  if (inMath) out += '\\)'; // auto-close any unclosed region
  return out;
};

// Replace <math>...</math> that aren't real MathML with TeX delimiters.
// Keep genuine MathML intact. Guard against SSR.
export function normalizeMathPlaceholders(html: string): string {
  if (!html || typeof window === 'undefined') return html;

  const parser = new DOMParser();
  const doc = parser.parseFromString(`<div id="__root">${html}</div>`, 'text/html');
  const root = doc.getElementById('__root') as HTMLElement;
  if (!root) return html;

  const MATHML_TAG_RE = /<(mi|mo|mn|mrow|msup|msub|mfrac|msqrt|mstyle|mtext|munderover|mover|munder|mtable|mtr|mtd)\b/i;

  for (const m of Array.from(root.querySelectorAll('math'))) {
    const inner = (m as HTMLElement).innerHTML.trim();
    const isRealMathML = MATHML_TAG_RE.test(inner);
    if (isRealMathML) continue;

    const tex = (m.textContent || '').trim();
    const displayAttr = (m.getAttribute('display') || '').toLowerCase();
    const display = displayAttr === 'block' || /\n/.test(tex) || /\\begin|\\frac|=/.test(tex);

    m.replaceWith(doc.createTextNode(display ? `$$ ${tex} $$` : `\\(${tex}\\)`));
  }
  return root.innerHTML;
}
