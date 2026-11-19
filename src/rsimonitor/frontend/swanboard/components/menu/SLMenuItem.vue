<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <MenuItem :as="as" class="p-3 hover:bg-higher cursor-pointer">
    <slot></slot>
  </MenuItem>
</template>

<script setup>
/**
 * SLMenuItem - 菜单项组件，封装headlessui的MenuItem组件
 */
import { MenuItem } from '@headlessui/vue'

defineProps({
  as: {
    type: String,
    default: 'div'
  }
})
</script>
