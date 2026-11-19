<template><span class="icon" aria-hidden="true" v-html="svg"></span></template>
<script setup>
import {computed} from 'vue';
const props = defineProps({name: {type: String, required: true}});
// lucide.min.js is a separately served, licensed asset; icon markup is static library data.
const pascal = name => name.replace(/(^|-)([a-z0-9])/g, (_, __, letter) => letter.toUpperCase());
const svg = computed(() => {
  const node = window.lucide?.icons?.[pascal(props.name)];
  return node && window.lucide.createElement ? window.lucide.createElement(node).outerHTML : '';
});
</script>
