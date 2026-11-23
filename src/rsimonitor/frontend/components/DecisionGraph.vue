<template>
  <div :class="{'trajectory-placeholder': expanded}">
    <Teleport :to="dialogBody" :disabled="!expanded">
      <div class="trajectory-shell" :class="{'in-dialog': expanded}">
        <div ref="host" class="flow-map" tabindex="0" :aria-label="graphTitle" @keydown="shortcut"></div>
        <div class="trajectory-jump" role="group" :aria-label="t('快速定位')">
          <button type="button" :class="{active: mode === 'root'}" :title="t('回到根节点') + ' (Home)'" @click="jump('root')">{{ t('根节点') }}</button>
          <button type="button" :disabled="!forks.length" :title="t(forks.length ? '下一个分叉' : '这次运行没有分叉') + ' (F)'" @click="jump('fork')">
            {{ t('下一个分叉') }}<span v-if="forks.length" class="count">{{ forks.length }}</span></button>
          <button type="button" :class="{active: mode === 'deepest'}" :title="t('最深的节点') + ' (End)'" @click="jump('deepest')">{{ t('最深节点') }}</button>
        </div>
        <div class="trajectory-tools">
          <button v-if="!expanded" ref="expandTrigger" type="button" aria-haspopup="dialog"
            :title="t('全屏查看')" :aria-label="t('全屏查看')" @click="openGraphDialog"><Icon name="maximize-2" /></button>
          <button type="button" :title="t('放大')" :aria-label="t('放大')" @click="graph?.zoomBy(1.25)">+</button>
          <button type="button" :title="t('缩小')" :aria-label="t('缩小')" @click="graph?.zoomBy(0.8)">−</button>
          <button type="button" :title="t('重置布局')" :aria-label="t('重置布局')" @click="graph?.reset()">↺</button>
        </div>
      </div>
    </Teleport>
  </div>
  <dialog ref="dialog" class="trajectory-dialog" :aria-label="graphTitle" @close="onDialogClose">
    <header>
      <h2>{{ graphTitle }}</h2>
      <div class="trajectory-dialog-actions">
        <button type="button" :title="t('适应画布')" :aria-label="t('适应画布')" @click="fitAll"><Icon name="scan" /></button>
        <button ref="closeTrigger" type="button" :title="t('关闭全屏')" :aria-label="t('关闭全屏')" @click="dialog?.close()"><Icon name="x" /></button>
      </div>
    </header>
    <div ref="dialogBody" class="trajectory-dialog-body"></div>
  </dialog>
</template>
<script setup>
/** Dependency/adoption graph of reported decisions. Layout and drawing live in decisions.js (framework-free). */
import {computed, nextTick, ref, watch, onMounted, onBeforeUnmount} from 'vue';
import RSIDecisions from '../app/decisions.js';
import {t, language} from '../app/i18n.js';
import {stageTitle} from '../app/insight.js';
import Icon from './Icon.vue';
const props = defineProps({runId: {type: String, required: true}, rows: {type: Array, required: true},
  selected: {type: String, default: ''}, lineage: Object});
const emit = defineEmits(['select']);
const host = ref(null), graph = ref(null);
const dialog = ref(null), dialogBody = ref(null), expandTrigger = ref(null), closeTrigger = ref(null);
const expanded = ref(false);
const graphTitle = computed(() => t(props.lineage ? '版本谱系' : '决策图'));
// Node positions, camera and the followed target survive snapshot refreshes and page switches, per run.
const views = new Map();
const EDGE_LABELS = {
  sequence: '时间顺序（非因果）', dependency: '声明的依赖', root: '布局锚点（非依赖）',
  adoption: '已确认采纳', source: '来源引用（非依赖或采纳）',
};
// decisions.js orders by `timestamp`; current producers report `started_at`.
const rows = computed(() => props.rows.map(row => row.timestamp ? row : {...row, timestamp: row.started_at}));
const model = computed(() => props.lineage || RSIDecisions.layout(rows.value));
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
  if (!views.has(props.runId)) {
    const hasMainline = model.value.nodes.some(node => node.onMainline);
    views.set(props.runId, {follow: hasMainline ? null : 'deepest', autoFit: hasMainline});
  }
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
    rows: rows.value, model: model.value, view: state, selected: props.selected || null,
    height: 300, responsiveHeight: true, compact: true,
    startLabel: t('开始'), label: row => row.label || stageTitle(row) || t('未命名决策'),
    edgeLabel: kind => t(props.lineage && kind === 'dependency' ? '派生关系' : EDGE_LABELS[kind]),
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
async function openGraphDialog() {
  if (!dialog.value || !dialogBody.value || dialog.value.open) return;
  dialog.value.showModal();
  expanded.value = true;
  await nextTick();
  graph.value?.resizePreservingScale();
  closeTrigger.value?.focus({preventScroll: true});
}
async function onDialogClose() {
  expanded.value = false;
  await nextTick();
  graph.value?.resizePreservingScale();
  expandTrigger.value?.focus({preventScroll: true});
}
function shortcut(event) {
  const kind = {Home: 'root', End: 'deepest', f: 'fork', F: 'fork'}[event.key];
  if (!kind || event.metaKey || event.ctrlKey || event.altKey) return;
  event.preventDefault();
  jump(kind);
}
onMounted(draw);
watch(() => [props.rows, props.selected, props.runId, props.lineage, language.value], draw);
onBeforeUnmount(() => graph.value?.dispose());
</script>

<style scoped>
.trajectory-placeholder { min-height: 302px; }
.trajectory-dialog {
  position: fixed; inset: 0; width: 100%; max-width: none; height: 100dvh; max-height: none;
  margin: 0; padding: 0; border: 0; overflow: hidden; background: var(--surface); color: var(--ink);
}
.trajectory-dialog[open] { display: flex; flex-direction: column; }
.trajectory-dialog header {
  display: flex; align-items: center; justify-content: space-between; min-height: 52px;
  padding: 6px 16px; border-bottom: 1px solid var(--line);
}
.trajectory-dialog h2 { margin: 0; font-size: 1rem; }
.trajectory-dialog-actions { display: flex; gap: 4px; }
.trajectory-dialog-actions button {
  display: grid; place-items: center; width: 36px; height: 36px; border-radius: 6px; color: var(--muted);
}
.trajectory-dialog-actions button:hover { background: var(--hover); color: var(--ink); }
.trajectory-tools .icon, .trajectory-dialog-actions .icon { width: 16px; height: 16px; vertical-align: middle; }
.trajectory-dialog-body { flex: 1; min-height: 0; }
.trajectory-shell.in-dialog { height: 100%; margin: 0; border: 0; border-radius: 0; }
.in-dialog .flow-map { height: 100%; }
@media (max-width: 480px) {
  .trajectory-jump { bottom: 52px; }
}
</style>
