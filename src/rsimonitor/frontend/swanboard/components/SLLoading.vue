<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <div
    class="relative flex items-center justify-center rounded-full overflow-hidden animate-spin"
    :class="size ? `w-${size} h-${size}` : 'w-8 h-8'"
  >
    <div class="absolute w-full h-full rounded-full border-4 border-t-positive-default border-positive-dimmest"></div>
    <div class="w-2 h-2 z-10 rounded-full bg-positive-default animate-ping"></div>
  </div>
</template>

<script setup>
/**
 * @description: 加载动画
 * @file: SLLoading.vue
 * @since: 2023-12-12 17:35:53
 **/

defineProps({
  size: {
    type: String,
    default: '8'
  }
})
</script>

<style lang="scss" scoped></style>
