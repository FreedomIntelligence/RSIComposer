<template><button class="button" type="button"><Icon v-if="icon" :name="icon" />{{ label }}</button></template>
<script setup>
import Icon from './Icon.vue';
defineProps({label: {type: String, required: true}, icon: {type: String, default: ''}});
</script>
