<template>
  <SectionTitle :title="t('产物')" :count="records.length" />
  <DataTable :headers="[t('产物'), t('类型'), t('阶段'), t('模型')]" :empty="t('这次运行没有上报产物')"
    :rows="records.map(row => [row.artifact_id, row.kind, stageLabel(run, row.call_id || row.decision_id), row.model])" />
</template>
<script setup>
import {computed} from 'vue';
import SectionTitle from '../components/SectionTitle.vue';
import DataTable from '../components/DataTable.vue';
import {t} from '../app/i18n.js';
import {rowsOf, stageLabel} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const records = computed(() => rowsOf(props.run.artifacts));
</script>
