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

const PROSE_WORD_ALLOWLIST = new Set([
  'dx',
  'dy',
  'dt',
  'du',
  'dv',
  'sin',
  'cos',
  'tan',
  'sec',
  'csc',
  'cot',
  'log',
  'ln',
  'lim',
  'det',
  'begin',
  'end',
  'matrix',
  'bmatrix',
  'pmatrix',
  'vmatrix',
  'smallmatrix',
  'array',
  'cases',
  'aligned',
  'align',
  'gathered',
  'split',
]);

export const looksLikeStandaloneTeX = (s: string): boolean => {
  const t = normalizeEscapedTeX(s).trim();
  if (!t || looksLikeHTML(t) || hasTeXDelimiters(t)) return false;

  const startsWithCommand = /^\\[A-Za-z]+/.test(t);
  const hasMathSyntax = /\\[A-Za-z]+|[=^_]|\\begin\{/.test(t);
  if (!startsWithCommand && !hasMathSyntax) return false;

  const proseWords = (t.replace(/\\[A-Za-z]+/g, ' ').match(/[A-Za-z]{2,}/g) ?? [])
    .map(word => word.toLowerCase())
    .filter(word => !PROSE_WORD_ALLOWLIST.has(word));

  return startsWithCommand || proseWords.length === 0;
};

export const ensureDelimiters = (s: string) => {
  const t = normalizeEscapedTeX(s).trim();
  if (!t) return t;
  if (hasTeXDelimiters(t)) return t;
  const display = t.includes('\n') || /\\begin|\\frac|\\int|\\lim|=/.test(t);
  return display ? `$$ ${t} $$` : `\\(${t}\\)`;
};

const protectDelimitedMath = (text: string): { text: string; restore: (value: string) => string } => {
  const regions: string[] = [];
  const protectedText = text.replace(
    /\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\$[\s\S]*?\$/g,
    match => {
      const token = `@@MATH_REGION_${regions.length}@@`;
      regions.push(match);
      return token;
    },
  );

  return {
    text: protectedText,
    restore: value => value.replace(/@@MATH_REGION_(\d+)@@/g, (_match, index) => regions[Number(index)] ?? ''),
  };
};

const isWordChar = (ch: string | undefined): boolean => Boolean(ch && /[A-Za-z]/.test(ch));
const isSingleLetterVariable = (word: string): boolean => /^[A-Za-z]$/.test(word);
const isAllowedMathWord = (word: string): boolean =>
  PROSE_WORD_ALLOWLIST.has(word.toLowerCase()) || isSingleLetterVariable(word);

const hasNearbyMathSyntax = (text: string, index: number): boolean => {
  const nearby = text.slice(index, Math.min(text.length, index + 18));
  return /[=^_+\-*/]|\\[A-Za-z]|\{|\}/.test(nearby);
};

const isValidMathSpan = (span: string): boolean => {
  const trimmed = span.trim();
  if (!trimmed) return false;
  if (/^@@MATH_REGION_\d+@@$/.test(trimmed)) return false;
  if (!/[\\=^_+\-*/]|\d[A-Za-z]|[A-Za-z]\d/.test(trimmed)) return false;

  const proseWords = (trimmed.replace(/\\[A-Za-z]+/g, ' ').match(/[A-Za-z]{2,}/g) ?? [])
    .map(word => word.toLowerCase())
    .filter(word => !PROSE_WORD_ALLOWLIST.has(word));

  return proseWords.length === 0;
};

const findMathSpanEnd = (text: string, start: number): number => {
  let i = start;

  while (i < text.length) {
    const ch = text[i];

    if (ch === '\\') {
      const command = text.slice(i).match(/^\\[A-Za-z]+/);
      if (command) {
        i += command[0].length;
        continue;
      }
      i += 1;
      continue;
    }

    if (/[0-9]/.test(ch)) {
      i += 1;
      continue;
    }

    if (/[{}()[\]^_=+\-*/|.,&\s]/.test(ch)) {
      i += 1;
      continue;
    }

    if (isWordChar(ch)) {
      const match = text.slice(i).match(/^[A-Za-z]+/);
      const word = match?.[0] ?? ch;
      if (isAllowedMathWord(word)) {
        i += word.length;
        continue;
      }
      break;
    }

    break;
  }

  let end = i;
  while (end > start && /[\s,.;:?]/.test(text[end - 1])) end -= 1;
  return end;
};

export const autoDelimitEmbeddedTeX = (text: string): string => {
  const { text: protectedText, restore } = protectDelimitedMath(text);
  let out = '';
  let i = 0;

  while (i < protectedText.length) {
    const ch = protectedText[i];
    const startsCommand = ch === '\\' && /^\\[A-Za-z]+/.test(protectedText.slice(i));
    const previousIsWord = isWordChar(protectedText[i - 1]);
    const nextIsWord = isWordChar(protectedText[i + 1]);
    const startsVariableExpression =
      !previousIsWord &&
      !nextIsWord &&
      isWordChar(ch) &&
      isSingleLetterVariable(ch) &&
      hasNearbyMathSyntax(protectedText, i);

    if (!startsCommand && !startsVariableExpression) {
      out += ch;
      i += 1;
      continue;
    }

    const end = findMathSpanEnd(protectedText, i);
    const span = protectedText.slice(i, end);

    if (end > i && isValidMathSpan(span)) {
      out += `\\(${span.trim()}\\)`;
      i = end;
      continue;
    }

    out += ch;
    i += 1;
  }

  return restore(out);
};

export const normalizeMathForRendering = (text: string, mathOnly = false): string => {
  const normalized = normalizeEscapedTeX(text);
  if (mathOnly || looksLikeStandaloneTeX(normalized)) {
    return ensureDelimiters(normalized);
  }
  return autoDelimitEmbeddedTeX(normalizeDollarDelimiters(normalized));
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
