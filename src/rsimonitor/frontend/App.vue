<template>
  <aside class="sidebar" :class="{open: menuOpen}">
    <a class="wordmark" href="#experiments">RSISys<span>.</span></a>
    <nav :aria-label="t('导航')">
      <a href="#experiments" :class="{active: store.page === 'experiments'}"><Icon name="layers-2" />{{ t('实验') }}<span class="count">{{ store.runs.length }}</span></a>
      <a href="#compare" :class="{active: store.page === 'compare'}"><Icon name="git-compare-arrows" />{{ t('对比') }}</a>
    </nav>
    <template v-if="run">
      <button class="run-switch" type="button" :title="t('切换实验')" @click="store.modal = {title: t('切换实验'), kind: 'runs'}">
        <span>{{ runLabel(run) }}</span><Icon name="chevrons-up-down" />
      </button>
      <nav :aria-label="t('当前实验')">
        <a v-for="page in pages" :key="page" :href="'#' + page" :class="{active: store.page === page}"><Icon :name="ICONS[page]" />{{ t(PAGE_LABELS[page]) }}</a>
      </nav>
    </template>
    <div class="sidebar-foot">
      <span class="sync" :class="{bad: store.error}" :title="store.error ? t(store.error.key, store.error.params) : ''"><i></i>{{ t(store.error ? '同步异常' : '实时') }}</span>
      <button type="button" class="ghost" :aria-label="t('界面语言')" @click="setLanguage(language === 'zh' ? 'en' : 'zh')">{{ language === 'zh' ? 'EN' : '中' }}</button>
      <button type="button" class="ghost" :title="t('立即同步')" :aria-label="t('立即同步')" @click="refresh"><Icon name="rotate-cw" /></button>
    </div>
  </aside>
  <button class="menu-button ghost" type="button" :aria-label="t('导航')" @click="menuOpen = !menuOpen"><Icon name="menu" /></button>
  <main id="main" tabindex="-1">
    <component :is="INDEX[store.page]" v-if="INDEX[store.page]" />
    <template v-else-if="run">
      <header class="page-head run-head">
        <div>
          <div class="title-row"><h1>{{ runLabel(run) }}</h1><StatusBadge :status="run.status" /><span v-if="run.simulated === true" class="tag">{{ t('模拟') }}</span></div>
          <p v-if="run.objective" class="objective">{{ run.objective }}</p>
          <p class="meta">{{ meta }}</p>
        </div>
        <ActionButton :label="t('导出')" icon="download" @click="download(`${run.run_id}.json`, run)" />
      </header>
      <KeepAlive include="Metrics"><component :is="RUN[store.page]" :key="run.run_id + store.page" :run="run" /></KeepAlive>
    </template>
    <p v-else class="empty-state">{{ t(store.runId ? '这个实验还没有快照' : '还没有实验。运行 rsisys run 后会出现在这里。') }}</p>
  </main>
  <div v-if="store.toast" class="toast" role="status">{{ t(store.toast) }}</div>
  <dialog ref="dialog" @close="store.modal = null" @click="backdrop">
    <template v-if="store.modal">
      <header><h2>{{ store.modal.title }}</h2><button class="ghost" type="button" :aria-label="t('关闭')" @click="store.modal = null"><Icon name="x" /></button></header>
      <div v-if="store.modal.kind === 'runs'" class="switch-list">
        <button v-for="row in store.runs" :key="row.run_id" type="button" :class="{active: row.run_id === store.runId}" @click="selectRun(row.run_id)">
          <span>{{ runLabel(row) }}</span><StatusBadge :status="row.status" /></button>
      </div>
      <pre v-else class="code-block">{{ JSON.stringify(store.modal.payload, null, 2) }}</pre>
    </template>
  </dialog>
</template>
<script setup>
import {computed, ref, watch} from 'vue';
import Icon from './components/Icon.vue';
import StatusBadge from './components/StatusBadge.vue';
import ActionButton from './components/ActionButton.vue';
import Experiments from './pages/Experiments.vue';
import Compare from './pages/Compare.vue';
import Overview from './pages/Overview.vue';
import Trajectory from './pages/Trajectory.vue';
import Metrics from './pages/Metrics.vue';
import Samples from './pages/Samples.vue';
import Artifacts from './pages/Artifacts.vue';
import Resources from './pages/Resources.vue';
import Logs from './pages/Logs.vue';
import {store, snapshot, refresh, selectRun, download, RUN_PAGES, PAGE_LABELS} from './app/store.js';
import {t, language, setLanguage, dateText, durationText} from './app/i18n.js';
import {rowsOf, runLabel} from './app/insight.js';
const INDEX = {experiments: Experiments, compare: Compare};
const RUN = {overview: Overview, trajectory: Trajectory, metrics: Metrics, samples: Samples, artifacts: Artifacts, resources: Resources, logs: Logs};
const ICONS = {overview: 'layout-dashboard', trajectory: 'route', metrics: 'chart-no-axes-combined', samples: 'list-checks',
  artifacts: 'package', resources: 'cpu', logs: 'terminal'};
const run = computed(() => snapshot.value);
// Pages a run has nothing to show on are left out of the navigation instead of leading to an empty screen.
const HAS_DATA = {
  samples: value => rowsOf(value.samples).length, artifacts: value => rowsOf(value.artifacts).length,
  resources: value => Object.keys(value.resources || {}).length,
};
const pages = computed(() => RUN_PAGES.filter(page => !HAS_DATA[page] || HAS_DATA[page](run.value) || store.page === page));
// One quiet line: when it started, how long it ran, and the ID only when it differs from the name.
const meta = computed(() => [dateText(run.value.started_at), durationText(run.value.workflow?.elapsed_seconds),
  run.value.project?.project_id !== 'unassigned' ? run.value.project?.name : '', run.value.experiment_name !== run.value.run_id ? run.value.run_id : '']
  .filter(text => text && text !== '—').join('  ·  '));
const menuOpen = ref(false), dialog = ref(null);
watch(() => store.page, () => { menuOpen.value = false; });
watch(() => store.modal, modal => {
  if (modal && !dialog.value.open) dialog.value.showModal();
  else if (!modal && dialog.value.open) dialog.value.close();
}, {flush: 'post'});
function backdrop(event) {
  if (event.target !== dialog.value) return;
  const box = dialog.value.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) store.modal = null;
}
</script>
