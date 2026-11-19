<template>
  <div v-if="stages.length" class="table-wrap">
    <table class="stage-table">
      <thead><tr><th class="num">#</th><th>{{ t('阶段') }}</th><th>{{ t('结果') }}</th><th class="num">{{ t('耗时') }}</th><th>{{ t('状态') }}</th></tr></thead>
      <tbody>
        <tr v-for="stage in stages" :key="stage.id" :class="{selected: stage.id === selected}" tabindex="0"
          @click="$emit('select', stage.id)" @keydown.enter="$emit('select', stage.id)">
          <td class="num muted">{{ stage.index }}</td>
          <td class="strong">{{ stage.label }}</td>
          <td class="muted">{{ stageResult(run, stage.row) || '—' }}</td>
          <td class="num muted">{{ durationText(stageSeconds(stage.row)) }}</td>
          <td><StatusBadge :status="stage.row.status" /></td>
        </tr>
      </tbody>
    </table>
  </div>
  <p v-else class="empty-state">{{ t('还没有上报任何阶段') }}</p>
</template>
<script setup>
import {computed} from 'vue';
import StatusBadge from './StatusBadge.vue';
import {t, durationText} from '../app/i18n.js';
import {stagesOf, stageResult, stageSeconds} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}, selected: {type: String, default: ''}});
defineEmits(['select']);
const stages = computed(() => stagesOf(props.run));
</script>
