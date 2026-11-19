<template>
  <div class="trajectory-shell">
    <div ref="host" class="flow-map" tabindex="0" :aria-label="t('决策图')" @keydown="shortcut"></div>
    <div class="trajectory-jump" role="group" :aria-label="t('快速定位')">
      <button type="button" :class="{active: mode === 'root'}" :title="t('回到根节点') + ' (Home)'" @click="jump('root')">{{ t('根节点') }}</button>
      <button type="button" :disabled="!forks.length" :title="t(forks.length ? '下一个分叉' : '这次运行没有分叉') + ' (F)'" @click="jump('fork')">
        {{ t('下一个分叉') }}<span v-if="forks.length" class="count">{{ forks.length }}</span></button>
      <button type="button" :class="{active: mode === 'deepest'}" :title="t('最深的节点') + ' (End)'" @click="jump('deepest')">{{ t('最深节点') }}</button>
    </div>
    <div class="trajectory-tools">
      <button type="button" :title="t('适应画布')" :aria-label="t('适应画布')" @click="fitAll">⤢</button>
      <button type="button" :title="t('放大')" :aria-label="t('放大')" @click="graph?.zoomBy(1.25)">+</button>
      <button type="button" :title="t('缩小')" :aria-label="t('缩小')" @click="graph?.zoomBy(0.8)">−</button>
      <button type="button" :title="t('重置布局')" :aria-label="t('重置布局')" @click="graph?.reset()">↺</button>
    </div>
  </div>
</template>
<script setup>
/** Dependency/adoption graph of reported decisions. Layout and drawing live in decisions.js (framework-free). */
import {computed, ref, watch, onMounted, onBeforeUnmount} from 'vue';
import RSIDecisions from '../app/decisions.js';
import {t, language} from '../app/i18n.js';
import {stageTitle} from '../app/insight.js';
const props = defineProps({runId: {type: String, required: true}, rows: {type: Array, required: true}, selected: {type: String, default: ''}});
const emit = defineEmits(['select']);
const host = ref(null), graph = ref(null);
// Node positions, camera and the followed target survive snapshot refreshes and page switches, per run.
const views = new Map();
const EDGE_LABELS = {
  sequence: '时间顺序（非因果）', dependency: '声明的依赖', root: '布局锚点（非依赖）',
  adoption: '已确认采纳', source: '来源引用（非依赖或采纳）',
};
// decisions.js orders by `timestamp`; current producers report `started_at`.
const rows = computed(() => props.rows.map(row => row.timestamp ? row : {...row, timestamp: row.started_at}));
const model = computed(() => RSIDecisions.layout(rows.value));
/** Depth of every node along structural edges, and where the graph branches. */
const shape = computed(() => {
  const children = new Map(model.value.nodes.map(node => [node.key, []]));
  for (const edge of model.value.edges) if (edge.kind !== 'source') children.get(edge.from)?.push(edge.to);
  const depth = new Map([['root', 0]]), queue = ['root'];
  while (queue.length) {
    const key = queue.shift();
    for (const child of children.get(key) || []) {
      // Longest path: a node reached again by a deeper route moves down.
      if ((depth.get(child) ?? -1) < depth.get(key) + 1 && depth.get(key) < model.value.nodes.length) {
        depth.set(child, depth.get(key) + 1);
        queue.push(child);
      }
    }
  }
  const order = model.value.nodes.map(node => node.key);
  const byDepth = (a, b) => (depth.get(a) ?? 0) - (depth.get(b) ?? 0) || order.indexOf(a) - order.indexOf(b);
  const forks = order.filter(key => (children.get(key) || []).length > 1).sort(byDepth);
  // Among equally deep nodes the most recently reported one is where the run currently is.
  const deepest = order.slice().sort((a, b) => -byDepth(a, b) || order.indexOf(b) - order.indexOf(a))[0] || 'root';
  return {forks, deepest};
});
const forks = computed(() => shape.value.forks);
const mode = ref('deepest');
let pending = false, forkIndex = -1;
function view() {
  // A run opens on its deepest node and keeps following it while the run grows.
  if (!views.has(props.runId)) views.set(props.runId, {follow: 'deepest', autoFit: false});
  return views.get(props.runId);
}
function target(kind) {
  return kind === 'root' ? 'root' : kind === 'deepest' ? shape.value.deepest : null;
}
function draw() {
  if (!host.value) return;
  const state = view();
  if (state.interacting || state.animating) { pending = true; return; }
  pending = false;
  // If the camera is no longer where a jump left it, the reader has panned: stop following.
  if (state.follow && state.landed && (state.camera?.x !== state.landed.x || state.camera?.y !== state.landed.y)) state.follow = null;
  mode.value = state.follow || '';
  if (state.follow) state.pendingFocus = target(state.follow);
  graph.value?.dispose();
  graph.value = RSIDecisions.render({
    rows: rows.value, model: model.value, view: state, selected: props.selected || null, height: 300, compact: true,
    startLabel: t('开始'), label: row => row.label || stageTitle(row) || t('未命名决策'), edgeLabel: kind => t(EDGE_LABELS[kind]),
    onSelect: (key, row) => { if (row) emit('select', row); },
    onIdle: () => { if (pending) draw(); },
  });
  host.value.replaceChildren(graph.value);
  // Two frames: the renderer centres the node after it has measured the real viewport.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (state.follow && state.camera) state.landed = {x: state.camera.x, y: state.camera.y};
  }));
}
function jump(kind) {
  const state = view();
  let key = target(kind);
  if (kind === 'fork') {
    if (!forks.value.length) return;
    forkIndex = (forkIndex + 1) % forks.value.length;
    key = forks.value[forkIndex];
  }
  state.follow = kind === 'fork' ? null : kind;
  mode.value = state.follow || '';
  graph.value?.focus(key);
  state.landed = state.camera ? {x: state.camera.x, y: state.camera.y} : null;
}
function fitAll() {
  view().follow = null;
  mode.value = '';
  graph.value?.fit();
}
function shortcut(event) {
  const kind = {Home: 'root', End: 'deepest', f: 'fork', F: 'fork'}[event.key];
  if (!kind || event.metaKey || event.ctrlKey || event.altKey) return;
  event.preventDefault();
  jump(kind);
}
onMounted(draw);
watch(() => [props.rows, props.selected, props.runId, language.value], draw);
onBeforeUnmount(() => graph.value?.dispose());
</script>
