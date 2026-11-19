<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <div
    class="flex items-center gap-2"
    :class="{ 'cursor-pointer': !disabled }"
    @click="!disabled && $emit('update:checked', !checked)"
  >
    <input type="checkbox" :checked="checked && !disabled" :disabled="disabled" />
    <span class="select-none truncate">{{ label }}</span>
  </div>
</template>

<script setup>
/**
 * @description: 选项框
 * @file: SLCheck.vue
 * @since: 2024-01-16 18:13:34
 **/

defineProps({
  label: {
    type: String,
    default: ''
  },
  checked: {
    type: Boolean,
    default: false
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

defineEmits(['update:checked'])
</script>

<style lang="scss" scoped>
input[type='checkbox'] {
  @apply border cursor-pointer w-4 h-4;
  &:disabled {
    @apply cursor-not-allowed;
  }
}
</style>
