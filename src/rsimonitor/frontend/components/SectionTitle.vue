<template>
  <div class="section-title"><h2>{{ title }}<span v-if="count !== null" class="count">{{ count }}</span></h2><div class="section-actions"><slot /></div></div>
</template>
<script setup>
defineProps({title: {type: String, required: true}, count: {type: [Number, String], default: null}});
</script>
