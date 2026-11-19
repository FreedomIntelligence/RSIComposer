<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <div class="sa-menu-items">
    <slot></slot>
  </div>
</template>

<script setup></script>

<style lang="scss" scoped>
.sa-menu-items {
  @apply w-full;
  &:not(:last-child) {
    @apply border-b pb-2;
  }
}
</style>
