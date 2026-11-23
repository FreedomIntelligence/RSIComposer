<template>
  <details v-if="content" class="version-content" :open="Boolean(version.parent_ids?.length)">
    <summary>{{ t(version.parent_ids?.length ? '版本改动' : '版本内容') }}</summary>
    <label v-if="content.parents.length > 1" class="parent-picker">
      {{ t('父版本') }}
      <select v-model="parentIndex">
        <option v-for="(item, index) in content.parents" :key="item.version_id" :value="index">
          {{ labels.get(item.version_id) || item.version_id.slice(0, 8) }}
        </option>
      </select>
    </label>
    <p v-if="parent && !comparable" class="muted">{{ t('父版本未发布内容') }}</p>
    <p v-else-if="comparable && !changes.length" class="muted">{{ t('与父版本内容相同') }}</p>
    <section v-for="change in changes" :key="change.name" class="component-change">
      <h3><code>{{ change.name }}</code></h3>
      <div class="change-text" :class="{comparison: comparable}">
        <div v-if="comparable">
          <h4>{{ t('改动前') }} · {{ labels.get(parent.version_id) || parent.version_id.slice(0, 8) }}</h4>
          <pre>{{ change.before ?? '∅' }}</pre>
        </div>
        <div>
          <h4>{{ comparable ? t('改动后') : t('版本内容') }} · {{ labels.get(version.version_id) }}</h4>
          <pre>{{ change.after ?? '∅' }}</pre>
        </div>
      </div>
    </section>
  </details>
  <p v-else-if="failed" class="muted version-content">{{ t('版本内容加载失败') }}</p>
</template>

<script setup>
import {computed, ref, watch} from 'vue';
import {t} from '../app/i18n.js';

const props = defineProps({runId: String, version: Object, labels: Map});
const content = ref(null), parentIndex = ref(0), failed = ref(false);
const parent = computed(() => content.value?.parents[parentIndex.value]);
const comparable = computed(() => Boolean(Object.keys(parent.value?.components || {}).length));
const changes = computed(() => {
  const before = parent.value?.components || {}, after = content.value?.components || {};
  return [...new Set([...Object.keys(after), ...Object.keys(before)])]
    .filter(name => !comparable.value || before[name] !== after[name])
    .map(name => ({name, before: Object.hasOwn(before, name) ? before[name] : undefined,
      after: Object.hasOwn(after, name) ? after[name] : undefined}));
});
watch(() => JSON.stringify([props.runId, props.version.version_id, props.version.parent_ids, props.version.content_available]), async (_, __, cleanup) => {
  const controller = new AbortController();
  cleanup(() => controller.abort());
  failed.value = false;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(props.runId)}/versions/${encodeURIComponent(props.version.version_id)}`,
      {signal: controller.signal, cache: 'no-store'});
    if (!response.ok && response.status !== 404) throw new Error(`HTTP ${response.status}`);
    const next = response.ok ? await response.json() : null;
    if (!controller.signal.aborted) { content.value = next; parentIndex.value = 0; }
  } catch {
    if (!controller.signal.aborted) { content.value = null; failed.value = true; }
  }
}, {immediate: true});
</script>

<style scoped>
.version-content { flex: 0 0 100%; min-width: 0; }
summary { cursor: pointer; font-weight: 600; }
.parent-picker { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.parent-picker select { max-width: 100%; color: inherit; background: var(--surface, #fff); }
.component-change { margin-top: 12px; }
.component-change h3, .component-change h4 { margin: 0 0 6px; font-size: .85rem; }
.change-text { display: grid; gap: 12px; }
.change-text.comparison { grid-template-columns: repeat(2, minmax(0, 1fr)); }
pre { margin: 0; padding: 10px 12px; max-height: 22rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere;
  border: 1px solid var(--line, #dce2e9); border-radius: 6px; font-size: .83rem; line-height: 1.6; }
@media (max-width: 760px) { .change-text.comparison { grid-template-columns: minmax(0, 1fr); } }
</style>
