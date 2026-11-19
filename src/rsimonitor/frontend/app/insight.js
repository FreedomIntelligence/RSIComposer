/** Derive what to show from reported data; no metric or split names are assumed. */
export const rowsOf = value => Array.isArray(value) ? value.filter(row => row && typeof row === 'object') : [];
export const decisionKey = row => row?.call_id || row?.decision_id || row?.directive_id || '';
export const runLabel = row => row?.experiment_name || row?.run_id || '';
export const scalarFields = object => Object.entries(object || {})
  .filter(([, value]) => ['number', 'boolean', 'string'].includes(typeof value));

const COUNT_KEY = /(^|[._/])(n|total|examples|count|eval_samples|num_samples|sample_count)$/;
const RATIO_KEY = /(^|[._/])(accuracy|macro_f1|f1|exact_match|pass_rate)$/;

/** {medqa: {accuracy: .3}} -> {'medqa.accuracy': .3}; non-numeric leaves are dropped. */
export function flattenMetrics(source, prefix = '', output = {}, depth = 0) {
  if (!source || typeof source !== 'object' || depth > 4) return output;
  for (const [key, value] of Object.entries(source)) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'number' && Number.isFinite(value)) output[name] = value;
    else if (value && typeof value === 'object') flattenMetrics(value, name, output, depth + 1);
  }
  return output;
}

export function metricDefinition(run, key) {
  const declared = run?.metric_definitions?.[key];
  if (declared) return declared;
  // Undeclared metrics stay plain numbers, except conventional [0, 1] quality ratios.
  return {label: key, unit: RATIO_KEY.test(key) ? 'ratio' : 'number', direction: RATIO_KEY.test(key) ? 'higher' : 'neutral'};
}

export function formatNumber(value, digits = 4) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  if (value !== 0 && (Math.abs(value) < 1e-4 || Math.abs(value) >= 1e9)) return value.toExponential(3);
  return value.toLocaleString('en-US', {maximumFractionDigits: digits});
}

export function formatMetric(value, definition) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const percent = definition?.unit === 'percent' || definition?.unit === 'ratio' && value >= 0 && value <= 1;
  if (percent) return `${formatNumber(value * 100, 2)}%`;
  if (definition?.unit === 'seconds') return `${formatNumber(value, 2)} s`;
  if (definition?.unit === 'tokens_per_second') return `${formatNumber(value, 1)} tok/s`;
  return formatNumber(value);
}

export function evaluationRows(run) {
  return rowsOf(run?.evaluations).map(row => ({...row, flat: flattenMetrics(row.metrics)}));
}

export function evaluationMetricKeys(runs) {
  const keys = new Set();
  for (const run of runs) for (const row of evaluationRows(run)) for (const key of Object.keys(row.flat)) keys.add(key);
  return [...keys].sort((a, b) => COUNT_KEY.test(a) - COUNT_KEY.test(b) || a.localeCompare(b));
}

/** The producer may declare `primary_metric`; otherwise prefer a metric with a declared direction. */
export function primaryMetric(run) {
  const keys = evaluationMetricKeys([run]);
  if (typeof run?.primary_metric === 'string' && keys.includes(run.primary_metric)) return run.primary_metric;
  const directed = keys.find(key => ['higher', 'lower'].includes(metricDefinition(run, key).direction));
  return directed || keys.find(key => !COUNT_KEY.test(key)) || keys[0] || '';
}

export const UNLABELLED = '';
export function splitsOf(runs) {
  const values = new Set();
  for (const run of runs) for (const row of rowsOf(run?.evaluations)) values.add(typeof row.split === 'string' ? row.split : UNLABELLED);
  return [...values].sort();
}

export function bestEvaluation(run, key, split) {
  const lower = metricDefinition(run, key).direction === 'lower';
  let best = null;
  for (const row of evaluationRows(run)) {
    if (split !== undefined && (typeof row.split === 'string' ? row.split : UNLABELLED) !== split) continue;
    const value = row.flat[key];
    if (typeof value !== 'number') continue;
    if (!best || (lower ? value < best.value : value > best.value)) best = {value, row};
  }
  return best;
}

export function sampleCount(row) {
  const flat = row.flat || flattenMetrics(row.metrics);
  const key = Object.keys(flat).find(name => COUNT_KEY.test(name));
  return key ? flat[key] : undefined;
}

/** The stage the page describes: the user's choice, else the live one merged with `current`. */
export function currentStage(run, chosenId) {
  const stages = rowsOf(run?.decisions), current = run?.current || {};
  const chosen = chosenId && stages.find(row => decisionKey(row) === chosenId);
  if (chosen) return chosen;
  const live = decisionKey(current) ? stages.find(row => decisionKey(row) === decisionKey(current)) : stages.at(-1);
  return live ? {...live, ...current} : current;
}

export const stageTitle = row => row?.operation || row?.action || row?.mode || row?.phase || row?.kind || '';

/** Curves for one stage, or for the whole run when no stage is given. */
export function metricSeries(run, stageId) {
  const groups = new Map();
  for (const point of rowsOf(run?.metric_points)) {
    const owner = decisionKey(point) || point.task_id;
    if (stageId && owner !== stageId) continue;
    if (typeof point.metric !== 'string' || !Number.isFinite(point.value)) continue;
    const axis = Number.isFinite(point.step) ? 'step' : Number.isFinite(point.elapsed_seconds) ? 'elapsed_seconds' : null;
    if (!axis) continue;
    const key = JSON.stringify([point.metric, axis]);
    if (!groups.has(key)) groups.set(key, new Map());
    // One line per reported series (GPU, worker...); otherwise per stage when plotted against its steps.
    const stage = point.series || (axis === 'step' ? owner || '' : '');
    const stages = groups.get(key);
    if (!stages.has(stage)) stages.set(stage, new Map());
    stages.get(stage).set(point[axis], point.value);
  }
  const stageNames = new Map(rowsOf(run?.decisions).map(row => [decisionKey(row), stageTitle(row)]));
  return Array.from(groups, ([key, stages]) => {
    const [metric, axis] = JSON.parse(key), definition = metricDefinition(run, metric);
    const sets = Array.from(stages, ([stage, values], index) => ({
      name: stageNames.has(stage) ? `${stageNames.get(stage)} · ${index + 1}` : stage || definition.label || metric,
      points: Array.from(values, ([x, y]) => ({x, y})).sort((a, b) => a.x - b.x),
    }));
    return {metric, axis, definition, sets, last: sets.at(-1)?.points.at(-1)?.y};
  });
}

/** The terminal "finish" decision closes the run; it is not a stage of work. */
const isTerminalMarker = row => (row.operation === 'finished' || row.kind === 'finished' || row.kind === 'finish')
  && !row.metrics && !row.job_id;

/** Stages worth showing, each with a readable label such as "math.train #2" instead of its UUID. */
export function stagesOf(run) {
  const rows = rowsOf(run?.decisions).filter(row => decisionKey(row) && !isTerminalMarker(row));
  const totals = new Map(), seen = new Map();
  for (const row of rows) totals.set(stageTitle(row), (totals.get(stageTitle(row)) || 0) + 1);
  return rows.map((row, index) => {
    const title = stageTitle(row) || '—', ordinal = (seen.get(title) || 0) + 1;
    seen.set(title, ordinal);
    return {row, id: decisionKey(row), index: index + 1, title, label: totals.get(title) > 1 ? `${title} #${ordinal}` : title};
  });
}

/** A log line usually repeats its stage name; show only what it adds ("started", a status...). */
export function logText(entry, title) {
  const text = entry.message || entry.summary || '';
  return title && text.startsWith(title + ' · ') ? text.slice(title.length + 3) : text;
}

export function stageLabel(run, id) {
  return stagesOf(run).find(stage => stage.id === id)?.label || '';
}

export function stageSeconds(row) {
  const start = Date.parse(row?.started_at), end = Date.parse(row?.finished_at);
  return Number.isFinite(start) && Number.isFinite(end) && end >= start ? (end - start) / 1000 : undefined;
}

/** One-line result of a stage: its first few reported metrics, formatted. */
export function stageResult(run, row, limit = 2) {
  return Object.entries(flattenMetrics(row?.metrics)).slice(0, limit)
    .map(([key, value]) => `${metricDefinition(run, key).label || key} ${formatMetric(value, metricDefinition(run, key))}`).join(' · ');
}

/** The primary metric across evaluations, in order: [{x, y, row}]. */
export function evaluationTrend(run, key) {
  return evaluationRows(run).filter(row => typeof row.flat[key] === 'number').map((row, index) => ({x: index + 1, y: row.flat[key], row}));
}
