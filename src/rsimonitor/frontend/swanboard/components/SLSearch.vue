<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <div class="swan-search" :class="{ 'icon-reverse': reverse }">
    <SLIcon class="swan-icon" icon="search" />
    <input
      type="text"
      v-model="value"
      class="focus:border-primary-default hover:border-primary-dimmer"
      :placeholder="placeholder"
      @input="input"
      @keydown.enter="$emit('search', value)"
    />
  </div>
</template>

<script setup>
/**
 * @description: 搜索
 * @file: SLSearch.vue
 * @since: 2024-01-08 16:36:14
 **/

import { ref } from 'vue'
import SLIcon from './SLIcon.vue'
import { debounce } from '@swanlab-vue/utils/common'

const emits = defineEmits(['input', 'search', 'update:modelValue'])

const props = defineProps({
  delay: {
    type: String,
    default: '0'
  },
  placeholder: {
    type: String,
    required: true
  },
  modelValue: {
    type: String,
    default: ''
  },
  reverse: {
    type: Boolean,
    default: false
  }
})

const value = ref(props.modelValue || '')

const input = debounce(() => {
  emits('input', value.value)
  emits('update:modelValue', value.value)
}, props.delay)
</script>

<style lang="scss" scoped>
.swan-search {
  @apply relative w-full rounded-lg;
  input {
    @apply block w-full outline-none text-sm text-dimmer border rounded-lg py-1.5 transition-all;
    padding-inline: 32px 12px;
  }

  .swan-icon {
    position: absolute;
    inset-block: 0;
    inset-inline-start: 10px;
    width: 16px;
    height: 16px;
    margin-block: auto;
    pointer-events: none;
  }
}

.icon-reverse {
  input {
    padding-inline: 12px 32px;
  }

  .swan-icon {
    inset-inline-start: auto;
    inset-inline-end: 10px;
  }
}
</style>
