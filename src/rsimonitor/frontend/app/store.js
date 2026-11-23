/** Single client-side state and sync loop for the monitor; every page reads from here. */
import {reactive, shallowRef} from 'vue';

export const RUN_PAGES = ['overview', 'trajectory', 'metrics', 'samples', 'artifacts', 'resources', 'logs'];
export const INDEX_PAGES = ['experiments', 'compare'];
export const PAGE_LABELS = {
  experiments: '实验', compare: '对比', overview: '总览', trajectory: '轨迹', metrics: '指标',
  samples: '样本', artifacts: '产物', resources: '资源', logs: '日志',
};
const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const POLL_MS = 2500, TIMEOUT_MS = 8000;

export const store = reactive({
  page: 'overview',
  runs: [],
  runId: '',
  error: null,        // {key, params} translated at render time
  decisionId: '',     // stage chosen by the user; empty follows the live stage
  evaluationId: '',
  stageLogs: '',
  compareIds: [],
});
/** Snapshots are large and replaced wholesale, so they stay out of deep reactivity. */
export const snapshot = shallowRef(null);
export const compared = shallowRef(new Map());

let newest = null, revision = 0, controller = null, timer = null;
const cache = new Map();  // url -> {etag, data}; the server answers 304 for unchanged snapshots

async function request(path, signal) {
  const known = cache.get(path);
  const response = await fetch(path, {signal, cache: 'no-store', headers: known ? {'If-None-Match': known.etag} : {}});
  if (response.status === 304 && known) return known.data;
  if (!response.ok) throw Object.assign(new Error('同步失败（HTTP {p0}）'), {params: {p0: response.status}});
  const data = await response.json(), etag = response.headers.get('ETag');
  if (etag) cache.set(path, {etag, data}); else cache.delete(path);
  return data;
}

const runUrl = id => `/api/runs/${encodeURIComponent(id)}`;
const started = row => Date.parse(row.started_at);
function validSnapshot(value, id) {
  return value?.schema_version === 1 && value.run_id === id;
}

/** Follow a newly started run unless the user is on a page where switching would lose their work. */
async function syncRuns(signal, current) {
  const data = await request('/api/runs', signal);
  if (current !== revision) return null;
  if (!Array.isArray(data.runs)) throw new Error('实验清单格式无效');
  const runs = data.runs.filter(row => row && RUN_ID.test(row.run_id));
  const latest = runs.filter(row => Number.isFinite(started(row)))
    .sort((a, b) => started(b) - started(a) || b.run_id.localeCompare(a.run_id))[0];
  const newer = latest && (!newest || started(latest) > started(newest)
    || started(latest) === started(newest) && latest.run_id.localeCompare(newest.run_id) > 0);
  const follow = store.page !== 'metrics';
  const id = newer && follow && (newest || !store.runId) ? latest.run_id
    : runs.some(row => row.run_id === store.runId) ? store.runId : latest?.run_id || runs[0]?.run_id || '';
  let next = null;
  if (id) {
    next = await request(runUrl(id), signal);
    if (current !== revision) return null;
    if (!validSnapshot(next, id)) throw new Error('快照与当前实验 ID 不匹配');
  }
  if (newer) newest = latest;
  return {runs, id, next};
}

async function syncCompared(signal) {
  const ids = store.compareIds.filter(id => store.runs.some(row => row.run_id === id));
  const results = await Promise.allSettled(ids.map(id => request(runUrl(id), signal)));
  const map = new Map(results.flatMap((result, i) =>
    result.status === 'fulfilled' && validSnapshot(result.value, ids[i]) ? [[ids[i], result.value]] : []));
  return {map, partial: map.size < ids.length};
}

export async function refresh() {
  clearTimeout(timer);
  controller?.abort();
  const own = controller = new AbortController(), current = ++revision;
  const timeout = setTimeout(() => own.abort(), TIMEOUT_MS);
  const active = () => controller === own;
  try {
    const result = await syncRuns(own.signal, current);
    if (!result || !active()) return;
    const switched = result.id !== store.runId;
    store.runs = result.runs;
    store.runId = result.id;
    if (snapshot.value !== result.next) snapshot.value = result.next;
    store.error = null;
    if (switched) {
      store.decisionId = store.evaluationId = store.stageLogs = '';
      if (RUN_PAGES.includes(store.page) && store.page !== 'overview') navigate('overview');
    }
    if (store.page === 'compare') {
      const {map, partial} = await syncCompared(own.signal);
      if (!active()) return;
      compared.value = map;
      if (partial) store.error = {key: '部分对比快照暂时无法同步'};
    }
  } catch (error) {
    if (active()) store.error = own.signal.aborted ? {key: '同步超时，正在重试'} : {key: error.message, params: error.params};
  } finally {
    clearTimeout(timeout);
    if (active()) {
      controller = null;
      if (!document.hidden) timer = setTimeout(refresh, POLL_MS);
    }
  }
}

export function selectRun(id) {
  if (!store.runs.some(row => row.run_id === id)) return;
  if (store.runId !== id) {
    store.runId = id;
    snapshot.value = null;
    revision++;
  }
  store.decisionId = store.evaluationId = store.stageLogs = '';
  store.error = null;
  navigate('overview');
  refresh();
}

export function toggleCompare(id, enabled) {
  store.compareIds = enabled ? [...new Set([...store.compareIds, id])] : store.compareIds.filter(value => value !== id);
  if (store.page === 'compare') refresh();
}

export function navigate(page) {
  if (location.hash !== `#${page}`) location.hash = page;
  else applyHash();
}

function applyHash() {
  const page = location.hash.slice(1);
  store.page = Object.hasOwn(PAGE_LABELS, page) ? page : 'overview';
}

export function start() {
  const requested = new URLSearchParams(location.search).get('run');
  if (requested && RUN_ID.test(requested)) store.runId = requested;
  applyHash();
  window.addEventListener('hashchange', () => {
    applyHash();
    window.scrollTo(0, 0);
    if (store.page === 'compare') refresh();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearTimeout(timer); else refresh();
  });
  refresh();
}
