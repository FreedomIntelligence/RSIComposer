<template>
  <article class="result-panel">
    <header><div><h3>{{ title }}</h3><p v-if="context">{{ context }}</p></div>
      <span v-if="change" class="result-change num" :class="tone">{{ change }}</span>
    </header>
    <dl class="result-rows">
      <div v-for="row in rows" :key="row.label" class="result-row" :class="{chosen: row.chosen}">
        <dt>{{ row.label }}</dt>
        <dd class="result-value num"><strong>{{ row.value }}</strong><small v-if="row.detail">{{ row.detail }}</small></dd>
        <dd v-if="maximum > 0 && Number.isFinite(row.amount) && row.amount >= 0 && row.amount <= maximum"
          class="result-track" aria-hidden="true"><span :style="{width: `${100 * row.amount / maximum}%`}" /></dd>
      </div>
    </dl>
    <div v-if="maximum > 0" class="result-scale num" aria-hidden="true"><span>{{ scaleStart }}</span><span>{{ scaleEnd }}</span></div>
  </article>
</template>
<script setup>
defineProps({title: String, context: String, rows: {type: Array, required: true}, change: String,
  tone: String, maximum: {type: Number, default: 0}, scaleStart: {type: String, default: '0'}, scaleEnd: String});
</script>
<style scoped>
.result-panel { min-width: 0; padding: 16px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: start; gap: 8px 16px; margin-bottom: 12px; }
h3 { color: var(--ink); font-size: 1rem; font-weight: 650; overflow-wrap: anywhere; }
header p, .result-value small, .result-scale { color: var(--muted); font-size: var(--text-small); }
.result-change { padding: 2px 8px; border-radius: 4px; background: var(--hover); color: var(--muted); font-size: var(--text-small); }
.result-change.good { background: var(--ok-soft); color: var(--ok); }
.result-change.bad { background: var(--bad-soft); color: var(--bad); }
.result-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 12px; align-items: baseline; padding: 8px 0; }
.result-row dt { overflow-wrap: anywhere; font-size: var(--text-ui); }
.result-value { display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: baseline; gap: 4px 8px; }
.result-value strong { font-size: 1rem; font-weight: 620; }
.chosen .result-value strong { color: var(--accent); font-size: 1.25rem; }
.result-track { grid-column: 1 / -1; height: 7px; border-radius: 2px; overflow: hidden; background: var(--hover); }
.result-track span { display: block; height: 100%; background: var(--faint); }
.chosen .result-track span { background: var(--accent); }
.result-scale { display: flex; justify-content: space-between; }
</style>
