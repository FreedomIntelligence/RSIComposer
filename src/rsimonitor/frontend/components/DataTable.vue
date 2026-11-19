<template>
  <div v-if="rows.length" class="table-wrap">
    <table>
      <thead><tr><th v-for="(header, column) in headers" :key="column" :class="{num: numeric.includes(column)}">{{ header }}</th></tr></thead>
      <tbody>
        <tr v-for="(row, index) in rows" :key="index">
          <td v-for="(cell, column) in row" :key="column" :class="{num: numeric.includes(column)}">{{ display(cell) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
  <p v-else class="empty-state">{{ empty }}</p>
</template>
<script setup>
defineProps({headers: {type: Array, required: true}, rows: {type: Array, required: true}, empty: {type: String, default: ''},
  numeric: {type: Array, default: () => []}});
const display = value => value === null || value === undefined || value === '' ? '—' : value;
</script>
