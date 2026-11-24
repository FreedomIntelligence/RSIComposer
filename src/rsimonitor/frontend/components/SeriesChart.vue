<template>
  <article class="chart-card">
    <header><div><h3>{{ title }}</h3><p v-if="context" class="muted series-context">{{ context }}</p></div>
      <span v-if="value" class="chart-value num">{{ value }}</span></header>
    <div ref="host" class="chart-wrap" role="img" :aria-label="title"></div>
    <footer v-if="axisLabel">{{ axisLabel }}</footer>
  </article>
</template>
<script setup>
/** The one small-chart implementation outside the metric workspace; same engine (G2Plot) as the workspace. */
import {ref, watch, onMounted, onBeforeUnmount} from 'vue';
import {Line} from '@antv/g2plot';
import {formatMetric} from '../app/insight.js';
const props = defineProps({
  title: {type: String, required: true}, value: {type: String, default: ''}, sets: {type: Array, required: true},
  definition: {type: Object, default: null}, axisLabel: {type: String, default: ''}, context: {type: String, default: ''},
});
const COLORS = ['#a85739', '#4f6850', '#63767b', '#8b754e', '#77668f', '#3b8791', '#b66b85'];
const host = ref(null);
let plot = null;
const rows = () => props.sets.flatMap(set => set.points.map(point => ({x: point.x, y: point.y, series: set.name})));
const label = value => formatMetric(Number(value), props.definition);
// Fit the axis to the data with a little headroom: a zero-based axis flattens small but real changes.
function range(data) {
  const values = data.map(row => row.y);
  if (!values.length) return {nice: true};
  const low = Math.min(...values), high = Math.max(...values), pad = (high - low || Math.abs(high) || 1) * 0.25;
  return {min: low >= 0 ? Math.max(0, low - pad) : low - pad, max: high + pad, nice: true};
}
function draw() {
  const data = rows();
  const integral = data.length > 0 && data.every(row => Number.isInteger(row.x));
  const span = integral ? Math.max(...data.map(row => row.x)) - Math.min(...data.map(row => row.x)) : 0;
  const axis = {style: {fontSize: 13, fill: '#656159'}};
  const options = {
    data, xField: 'x', yField: 'y', seriesField: 'series', color: COLORS, autoFit: true, animation: false,
    padding: 'auto', legend: props.sets.length > 1 ? {position: 'bottom'} : false, lineStyle: {lineWidth: 1.6},
    meta: {x: {type: 'linear', nice: true, tickInterval: integral ? Math.max(1, Math.ceil(span / 7)) : undefined}, y: range(data)},
    xAxis: {label: axis, line: {style: {stroke: '#e6e3dc'}}, tickLine: null},
    yAxis: {label: {...axis, formatter: label}, grid: {line: {style: {stroke: '#efede8'}}}},
    point: data.length < 60 ? {size: 2.5, shape: 'circle'} : false,
    tooltip: {formatter: datum => ({name: datum.series, value: label(datum.y)})},
  };
  if (plot) { plot.update(options); return; }
  plot = new Line(host.value, options);
  plot.render();
}
onMounted(draw);
watch(() => props.sets, draw);
onBeforeUnmount(() => { plot?.destroy(); plot = null; });
</script>
<style scoped>
.series-context { margin-top: 3px; font-size: var(--text-small); }
</style>
