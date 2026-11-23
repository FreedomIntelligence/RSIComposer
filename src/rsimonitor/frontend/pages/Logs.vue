<template>
  <SectionTitle :title="t('日志')" :count="shown.length">
    <button v-if="store.stageLogs" type="button" class="chip active" @click="store.stageLogs = ''">{{ stageLabel(run, store.stageLogs) }} ✕</button>
  </SectionTitle>
  <div v-if="shown.length" class="table-wrap">
    <table class="log-table">
      <tbody>
        <tr v-for="(entry, index) in shown" :key="index" :class="{bad: entry.level === 'ERROR'}">
          <td class="muted num nowrap">{{ timeText(entry.timestamp) }}</td>
          <td class="nowrap">{{ stageLabel(run, stageOf(entry)) || '—' }}</td>
          <td class="wide">{{ text(entry) }}</td>
          <td class="muted nowrap">{{ entry.level === 'INFO' ? '' : entry.level }}</td>
        </tr>
      </tbody>
    </table>
  </div>
  <p v-else class="empty-state">{{ t('还没有日志') }}</p>
  <section v-if="config.length" class="section">
    <SectionTitle :title="t('运行配置')" />
    <dl class="facts wide-facts"><template v-for="[key, value] in config" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template></dl>
  </section>
</template>
<script setup>
import {computed} from 'vue';
import SectionTitle from '../components/SectionTitle.vue';
import {store} from '../app/store.js';
import {t, timeText} from '../app/i18n.js';
import {rowsOf, scalarFields, stageLabel, stagesOf, logText} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const stageOf = entry => entry.call_id || entry.details?.call_id || entry.decision_id || entry.details?.decision_id;
const STATUS = {completed: '已完成', failed: '失败', cancelled: '已取消', running: '运行中'};
const titles = computed(() => new Map(stagesOf(props.run).map(stage => [stage.id, stage.title])));
// Lines that belong to the run's terminal marker rather than a stage say nothing a reader needs.
const records = computed(() => rowsOf(props.run.logs?.length ? props.run.logs : props.run.events)
  .filter(entry => !stageOf(entry) || titles.value.has(stageOf(entry))));
function text(entry) {
  const rest = logText(entry, titles.value.get(stageOf(entry)));
  return STATUS[rest] ? t(STATUS[rest]) : rest;
}
const shown = computed(() => store.stageLogs ? records.value.filter(entry => stageOf(entry) === store.stageLogs) : records.value);
const config = computed(() => scalarFields(props.run.config));
</script>
