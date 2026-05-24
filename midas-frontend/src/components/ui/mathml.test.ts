import {
  ensureDelimiters,
  normalizeDollarDelimiters,
  normalizeEscapedTeX,
  normalizeMathForRendering,
  autoDelimitEmbeddedTeX,
} from './mathml';

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

  it('wraps standalone raw TeX from uploaded problem text for rendering', () => {
    const input = String.raw`\int_0^2 (3x^{2} - 2x + 1)\, dx`;

    expect(normalizeMathForRendering(input))
      .toBe(String.raw`$$ \int_0^2 (3x^{2} - 2x + 1)\, dx $$`);
  });

  it('wraps embedded TeX commands inside prose', () => {
    const input = String.raw`Problem: If \sum_{n=0}^\infty \cos^{2n}\theta = 5, what is \cos 2\theta?`;

    expect(normalizeMathForRendering(input))
      .toBe(String.raw`Problem: If \(\sum_{n=0}^\infty \cos^{2n}\theta = 5\), what is \(\cos 2\theta\)?`);
  });

  it('wraps embedded algebra expressions inside prose', () => {
    const input = String.raw`Problem: The equation x^{2} + 2x = i has two complex solutions.`;

    expect(normalizeMathForRendering(input))
      .toBe(String.raw`Problem: The equation \(x^{2} + 2x = i\) has two complex solutions.`);
  });

  it('wraps matrix environments without splitting environment names', () => {
    const input = String.raw`Find the eigenvalues of the matrix A = \begin{bmatrix}4 & 1 \\ 2 & 3\end{bmatrix}`;

    expect(normalizeMathForRendering(input))
      .toBe(String.raw`Find the eigenvalues of the matrix \(A = \begin{bmatrix}4 & 1 \\ 2 & 3\end{bmatrix}\)`);
  });

  it('does not introduce delimiters inside existing matrix environments', () => {
    const input = String.raw`\(A = \begin{bmatrix}4 & 1 \\ 2 & 3\end{bmatrix}\)`;

    expect(normalizeMathForRendering(input)).toBe(input);
  });

  it('wraps simple inline equations inside prose', () => {
    const input = 'Solve x + 1 = 2.';

    expect(normalizeMathForRendering(input)).toBe(String.raw`Solve \(x + 1 = 2\).`);
  });

  it('does not wrap plain prose without math syntax', () => {
    const input = 'Select a problem above.';

    expect(normalizeMathForRendering(input)).toBe(input);
  });

  it('preserves existing delimiters when adding embedded raw TeX delimiters', () => {
    const input = String.raw`If \(x=1\), compute \cos \theta.`;

    expect(autoDelimitEmbeddedTeX(input))
      .toBe(String.raw`If \(x=1\), compute \(\cos \theta\).`);
  });
});
