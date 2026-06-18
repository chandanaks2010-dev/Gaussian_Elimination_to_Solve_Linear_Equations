/* ─── SolutionChart ──────────────────────────────────────────────────────── */
function SolutionChart({ variables, solution }) {
  const PAD_LEFT = 48;
  const PAD_RIGHT = 16;
  const PAD_TOP = 20;
  const PAD_BOTTOM = 48;
  const BAR_GAP = 10;
  const CHART_H = 180;

  const n = variables.length;
  const svgW = PAD_LEFT + PAD_RIGHT + n * (40 + BAR_GAP);
  const svgH = CHART_H + PAD_TOP + PAD_BOTTOM;

  // Axis extents — always include 0
  const maxVal = Math.max(0, ...solution);
  const minVal = Math.min(0, ...solution);
  const range = maxVal - minVal || 1;

  // Map a value to a Y pixel (0 at top, svgH at bottom)
  const toY = (v) => PAD_TOP + ((maxVal - v) / range) * CHART_H;
  const zeroY = toY(0);

  // Y-axis tick labels (5 ticks)
  const ticks = Array.from({ length: 5 }, (_, i) => {
    const val = minVal + (range / 4) * i;
    return { val, y: toY(val) };
  });

  const BAR_W = 40;
  const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

  return (
    <div className="chart-wrap">
      <h3 className="chart-title">Solution Values</h3>
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        width={svgW}
        height={svgH}
        aria-label="Bar chart of solution values"
        role="img"
      >
        {/* Y-axis grid lines & tick labels */}
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD_LEFT}
              x2={svgW - PAD_RIGHT}
              y1={t.y}
              y2={t.y}
              stroke="#e2e8f0"
              strokeWidth="1"
              strokeDasharray={i === 0 || i === 4 ? 'none' : '4 3'}
            />
            <text
              x={PAD_LEFT - 6}
              y={t.y + 4}
              textAnchor="end"
              fontSize="10"
              fill="#64748b"
            >
              {fmt(t.val)}
            </text>
          </g>
        ))}

        {/* Zero axis (bold) */}
        <line
          x1={PAD_LEFT}
          x2={svgW - PAD_RIGHT}
          y1={zeroY}
          y2={zeroY}
          stroke="#94a3b8"
          strokeWidth="1.5"
        />

        {/* Y-axis line */}
        <line
          x1={PAD_LEFT}
          x2={PAD_LEFT}
          y1={PAD_TOP}
          y2={PAD_TOP + CHART_H}
          stroke="#94a3b8"
          strokeWidth="1.5"
        />

        {/* Bars */}
        {solution.map((val, i) => {
          const barTop = val >= 0 ? toY(val) : zeroY;
          const barH = Math.abs(toY(val) - zeroY);
          const x = PAD_LEFT + i * (BAR_W + BAR_GAP) + BAR_GAP / 2;
          const positive = val >= 0;

          return (
            <g key={i}>
              <rect
                x={x}
                y={barTop}
                width={BAR_W}
                height={Math.max(barH, 2)}
                fill={positive ? '#22c55e' : '#f87171'}
                rx="4"
                opacity="0.85"
              />
              {/* Value label above/below bar */}
              <text
                x={x + BAR_W / 2}
                y={positive ? barTop - 5 : barTop + Math.max(barH, 2) + 13}
                textAnchor="middle"
                fontSize="10"
                fontWeight="700"
                fill={positive ? '#15803d' : '#dc2626'}
              >
                {fmt(val)}
              </text>
              {/* Variable label below axis */}
              <text
                x={x + BAR_W / 2}
                y={PAD_TOP + CHART_H + 18}
                textAnchor="middle"
                fontSize="11"
                fontWeight="600"
                fill="#0f172a"
              >
                {variables[i]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ─── MatrixView ─────────────────────────────────────────────────────────── */
function MatrixView({ matrix, n, pivotCell = null, highlightRows = [] }) {
  if (!matrix || matrix.length === 0) return null;

  const fmt = (v) => {
    const num = typeof v === 'number' ? v : parseFloat(v);
    if (Number.isInteger(num)) return String(num);
    const s = num.toFixed(4);
    // trim trailing zeros after decimal
    return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  };

  // Map of row index → highlight type ('pivot' | 'active')
  const rowHighlight = {};
  highlightRows.forEach(({ idx, type }) => { rowHighlight[idx] = type; });

  return (
    <div className="matrix-view">
      <table>
        <thead>
          <tr>
            {Array.from({ length: n }, (_, j) => (
              <th key={j}>
                x<sub>{j + 1}</sub>
              </th>
            ))}
            <th className="mv-bh">b</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i} className={rowHighlight[i] ? `mv-row-${rowHighlight[i]}` : ''}>
              {row.map((val, j) => {
                const isPivot = pivotCell && pivotCell.row === i && pivotCell.col === j;
                const cls = isPivot ? 'mv-pivot-cell' : j === n ? 'mv-bv' : '';
                return <td key={j} className={cls}>{fmt(val)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Step badge labels ───────────────────────────────────────────────────── */
const STEP_LABEL = {
  initial: 'Initial Matrix',
  warning: 'Ill-Conditioned Matrix',
  pivot: 'Pivot (Row Swap)',
  elimination: 'Row Elimination',
  upper_triangular: 'Upper Triangular',
  back_substitution: 'Back Substitution',
};

/* ─── SolutionDisplay ────────────────────────────────────────────────────── */
export default function SolutionDisplay({ result }) {
  const { solution, variables, steps, n } = result;

  return (
    <div className="solution-section">
      {/* Final answer box */}
      <div className="final-answer">
        <h2>Solution</h2>
        <div className="answer-grid">
          {variables.map((v, i) => (
            <div key={i} className="answer-item">
              <span className="av">{v}</span>
              <span className="aeq">=</span>
              <span className="aval">{solution[i]}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Solution bar chart */}
      <SolutionChart variables={variables} solution={solution} />

      {/* Step-by-step working */}
      <div className="steps-block">
        <h2>Step-by-Step Working</h2>
        {steps.map((step, idx) => (
          <div key={idx} className={`step-card sc-${step.type}`}>
            <div className="step-head">
              <span className={`badge b-${step.type}`}>
                {STEP_LABEL[step.type] ?? step.type}
              </span>
              <span className="step-desc">{step.description}</span>
            </div>

            {step.matrix && (
              <MatrixView
                matrix={step.matrix}
                n={n}
                pivotCell={
                  step.pivot_row != null && step.pivot_col != null
                    ? { row: step.pivot_row, col: step.pivot_col }
                    : null
                }
                highlightRows={
                  step.type === 'pivot'
                    ? [{ idx: step.pivot_row, type: 'pivot' }]
                    : step.type === 'elimination'
                    ? [
                        { idx: step.pivot_row, type: 'pivot' },
                        { idx: step.active_row, type: 'active' },
                      ]
                    : []
                }
              />
            )}

            {step.back_steps && (
              <div className="bs-list">
                {step.back_steps.map((bs, i) => (
                  <div key={i} className="bs-row">
                    <span className="bs-var">{bs.variable}</span>
                    <span className="bs-eq">=</span>
                    <span className="bs-val">{bs.value}</span>
                    <code className="bs-formula">{bs.description}</code>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
