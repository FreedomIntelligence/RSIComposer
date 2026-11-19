<template>
  <div v-if="stage" class="live-stage">
    <span class="pulse" aria-hidden="true"></span>
    <div class="live-text"><span class="eyebrow">{{ t('正在执行') }}</span><strong>{{ stage.label }}</strong></div>
    <div v-if="progress !== null" class="live-progress">
      <div class="progress-track"><div class="progress-fill" :style="{width: progress + '%'}"></div></div>
      <span class="num">{{ stage.row.step }} / {{ stage.row.total_steps }}</span>
    </div>
  </div>
</template>
<script setup>
/** Shown only while a run is executing: the stage in flight and, if reported, its progress. */
import {computed} from 'vue';
import {t} from '../app/i18n.js';
import {stagesOf} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const ACTIVE = ['running', 'starting', 'executing', 'waiting'];
const stage = computed(() => ACTIVE.includes(props.run.status)
  ? stagesOf(props.run).filter(item => ACTIVE.includes(item.row.status)).at(-1) : null);
const progress = computed(() => {
  const {step, total_steps: total} = stage.value?.row || {};
  return Number.isFinite(step) && Number.isFinite(total) && total > 0 ? Math.max(0, Math.min(100, 100 * step / total)) : null;
});
</script>
