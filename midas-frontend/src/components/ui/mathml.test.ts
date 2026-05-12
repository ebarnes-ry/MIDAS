import { ensureDelimiters, normalizeDollarDelimiters, normalizeEscapedTeX } from './mathml';

describe('mathml helpers', () => {
  it('normalizes doubled TeX delimiters and commands from model output', () => {
    const input = String.raw`The value is \\(\\frac{3}{5}\\).`;

    expect(normalizeEscapedTeX(input)).toBe(String.raw`The value is \(\frac{3}{5}\).`);
  });

  it('wraps bare LaTeX expressions so MathJax has a math region', () => {
    expect(ensureDelimiters(String.raw`\sum_{n=0}^{\infty} \cos^{2n}\theta = 5`))
      .toBe(String.raw`$$ \sum_{n=0}^{\infty} \cos^{2n}\theta = 5 $$`);
  });

  it('normalizes dollar delimiters after escaped TeX cleanup', () => {
    const input = String.raw`If $\\cos 2\\theta = \\frac{3}{5}$, then done.`;

    expect(normalizeDollarDelimiters(input))
      .toBe(String.raw`If \(\cos 2\theta = \frac{3}{5}\), then done.`);
  });

  it('preserves TeX line breaks that use a doubled backslash followed by whitespace', () => {
    const input = String.raw`a + b \\ c + d`;

    expect(normalizeEscapedTeX(input)).toBe(input);
  });
});
