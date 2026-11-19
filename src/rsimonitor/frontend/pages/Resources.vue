<template>
  <SectionTitle :title="t('资源')" />
  <dl v-if="rows.length" class="facts wide-facts"><template v-for="[key, value] in rows" :key="key"><dt>{{ key }}</dt><dd class="num">{{ value }}</dd></template></dl>
  <p v-else class="empty-state">{{ t('这次运行没有上报资源用量') }}</p>
</template>
<script setup>
import {computed} from 'vue';
import SectionTitle from '../components/SectionTitle.vue';
import {t} from '../app/i18n.js';
import {scalarFields} from '../app/insight.js';
const props = defineProps({run: {type: Object, required: true}});
const rows = computed(() => {
  const resources = props.run.resources || {};
  return [...scalarFields(resources).filter(([key]) => key !== 'decision_id' && key !== 'call_id'), ...scalarFields(resources.latest_execution_metrics)];
});
</script>
