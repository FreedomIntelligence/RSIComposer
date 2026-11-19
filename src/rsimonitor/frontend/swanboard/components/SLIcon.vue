<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <svg aria-hidden="true">
    <use :xlink:href="'#icon-' + icon"></use>
  </svg>
</template>

<script setup>
/**
 * @description: 对iconfont图标的使用进行封装
 * @file: SLIcon.vue
 * @since: 2023-12-04 16:07:51
 **/
import '@swanlab-vue/assets/iconfont.js'
defineProps({
  icon: {
    type: String,
    required: true
  }
})
</script>
