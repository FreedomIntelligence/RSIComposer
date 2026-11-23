<template>
  <ObservationEvents :run="run" category="quality" />
  <div class="rsi-metrics-ui"><Workspace :state="state" /></div>
  <details v-if="probes.length" class="technical">
    <summary>{{ t('观测触发器 · {p0}', {p0: probes.length}) }}</summary>
    <DataTable :headers="['Probe', t('触发事件'), t('采样间隔'), t('执行次数'), t('失败次数')]" :numeric="[2, 3, 4]"
      :rows="probes.map(p => [p.name, p.on, p.every, p.calls, p.failures])" />
  </details>
</template>
<script setup>
import {computed} from 'vue';
import Workspace from '../Workspace.vue';
import DataTable from '../components/DataTable.vue';
import ObservationEvents from '../components/ObservationEvents.vue';
import {store} from '../app/store.js';
import {t} from '../app/i18n.js';
import {rowsOf} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const state = computed(() => ({run: props.run, runs: store.runs}));
const probes = computed(() => rowsOf(props.run.probes));
</script>
