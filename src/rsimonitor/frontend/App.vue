<template>
  <header class="topbar">
    <button ref="navToggle" class="ghost" type="button" aria-controls="sidebar" :aria-expanded="sidebarOpen"
      :title="sidebarLabel" :aria-label="sidebarLabel" @click="toggleSidebar" @keydown.esc="closeDrawer(true)">
      <Icon :name="sidebarOpen ? 'panel-left-close' : 'panel-left-open'" />
    </button>
    <a class="wordmark" href="#experiments" @click="closeDrawer()">RSIComposer<span>.</span></a>
    <div class="topbar-actions">
      <button type="button" class="ghost" :title="t('立即同步')" :aria-label="t('立即同步')" @click="refresh"><Icon name="rotate-cw" /></button>
      <button type="button" class="ghost language-button" :title="t('界面语言')" :aria-label="t('界面语言')"
        @click="setLanguage(language === 'zh' ? 'en' : 'zh')"><Icon name="languages" />{{ language === 'zh' ? 'EN' : '中' }}</button>
      <a class="ghost" href="https://github.com/FreedomIntelligence/RSIComposer" target="_blank" rel="noopener noreferrer"
        aria-label="GitHub" title="GitHub"><Icon name="github" /></a>
    </div>
  </header>
  <div v-if="narrow && drawerOpen" class="sidebar-backdrop" aria-hidden="true" @click="closeDrawer()"></div>
  <aside id="sidebar" class="sidebar" :class="{open: sidebarOpen}" :inert="!sidebarOpen" @keydown.esc="closeDrawer(true)">
    <nav aria-labelledby="workspace-nav-title">
      <h2 id="workspace-nav-title" class="nav-label">{{ t('工作区') }}</h2>
      <a href="#experiments" :aria-current="store.page === 'experiments' ? 'page' : undefined" @click="closeDrawer()"><Icon name="layers-2" />{{ t('实验列表') }}<span class="count">{{ store.runs.length }}</span></a>
      <a href="#compare" :aria-current="store.page === 'compare' ? 'page' : undefined" @click="closeDrawer()"><Icon name="git-compare-arrows" />{{ t('对比') }}</a>
    </nav>
  </aside>
  <main id="main" tabindex="-1">
    <p v-if="store.error" class="sync-error" role="status">{{ t(store.error.key, store.error.params) }}</p>
    <component :is="INDEX[store.page]" v-if="INDEX[store.page]" />
    <template v-else-if="run">
      <header class="page-head run-head">
        <div>
          <a class="run-back" href="#experiments"><Icon name="arrow-left" />{{ t('实验列表') }}</a>
          <div class="title-row"><h1>{{ runLabel(run) }}</h1><StatusBadge :status="run.status" /><span v-if="run.simulated === true" class="tag">{{ t('模拟') }}</span></div>
          <p class="meta">{{ meta }}</p>
          <p v-if="run.objective" class="objective">{{ run.objective }}</p>
        </div>
      </header>
      <nav class="run-tabs" :aria-label="t('实验视图')">
        <a v-for="page in pages" :key="page" :href="'#' + page" :aria-current="store.page === page ? 'page' : undefined">
          <Icon :name="ICONS[page]" />{{ t(PAGE_LABELS[page]) }}
        </a>
      </nav>
      <KeepAlive include="Metrics"><component :is="RUN[store.page]" :key="run.run_id + store.page" :run="run" /></KeepAlive>
    </template>
    <p v-else class="empty-state">{{ t(store.runId ? '这个实验还没有快照' : '还没有实验。运行 rsicomposer run 后会出现在这里。') }}</p>
  </main>
</template>
<script setup>
import {computed, onMounted, onUnmounted, ref, watch} from 'vue';
import Icon from './components/Icon.vue';
import StatusBadge from './components/StatusBadge.vue';
import Experiments from './pages/Experiments.vue';
import Compare from './pages/Compare.vue';
import Overview from './pages/Overview.vue';
import Trajectory from './pages/Trajectory.vue';
import Metrics from './pages/Metrics.vue';
import Samples from './pages/Samples.vue';
import Artifacts from './pages/Artifacts.vue';
import Resources from './pages/Resources.vue';
import Logs from './pages/Logs.vue';
import {store, snapshot, refresh, RUN_PAGES, PAGE_LABELS} from './app/store.js';
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
// The readable experiment title stays separate from runtime metadata.
const meta = computed(() => [dateText(run.value.started_at), durationText(run.value.workflow?.elapsed_seconds),
  run.value.project?.project_id !== 'unassigned' ? run.value.project?.name : '']
  .filter(text => text && text !== '—').join('  ·  '));
const media = window.matchMedia('(max-width: 760px)');
const narrow = ref(media.matches), drawerOpen = ref(false), desktopOpen = ref(true);
const navToggle = ref(null);
const sidebarOpen = computed(() => narrow.value ? drawerOpen.value : desktopOpen.value);
const sidebarLabel = computed(() => sidebarOpen.value ? t('收起侧边栏') : t('展开侧边栏'));
const sidebarKey = 'rsicomposer.sidebar.collapsed';
try { desktopOpen.value = localStorage.getItem(sidebarKey) !== 'true'; } catch { /* Storage is optional. */ }
function syncWidth(event) {
  narrow.value = event.matches;
  drawerOpen.value = false;
}
onMounted(() => media.addEventListener('change', syncWidth));
onUnmounted(() => media.removeEventListener('change', syncWidth));
function toggleSidebar() {
  if (narrow.value) {
    drawerOpen.value = !drawerOpen.value;
  } else {
    desktopOpen.value = !desktopOpen.value;
    try { localStorage.setItem(sidebarKey, String(!desktopOpen.value)); } catch { /* Keep the in-memory preference. */ }
  }
}
function closeDrawer(restoreFocus = false) {
  if (!drawerOpen.value) return;
  drawerOpen.value = false;
  if (restoreFocus) navToggle.value?.focus();
}
watch(() => store.page, () => closeDrawer());
</script>
