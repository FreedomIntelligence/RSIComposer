<!--
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
-->
<template>
  <!-- 容器 -->
  <div class="charts-container" ref="chartsContainerRef">
    <!-- 控制栏 -->
    <div class="flex items-center">
      <!-- 关闭、展开按钮 -->
      <button class="flex items-center gap-1 relative" @click="handleNamespaceClick">
        <SLIcon icon="down" class="w-6 h-6 absolute -left-6" :class="{ '-rotate-90': !isExpand }" />
        {{ label }}
        <span class="rsi-group-count">{{ charts.length || 0 }}</span>
      </button>
      <!-- 中间其他操作区 -->
      <div class="grow"></div>
      <!-- 添加图表的按钮 -->
    </div>
    <!-- 图表网格 -->
    <div class="charts-slot" v-show="isExpand" ref="chartsSlotRef">
      <ChartContainer
        v-for="(chart, index) in charts"
        :key="chart.id"
        :chart="chart"
        :index="index"
        :isPinned="isPinned"
        :isHidden="isHidden"
        :ref="(el) => setChartRefList(el, index)"
        @pin="pin"
        @unpin="unpin"
        @hide="hide"
        @unhide="unhide"
      />
    </div>
  </div>
</template>

<script setup>
import SLIcon from '@swanlab-vue/components/SLIcon.vue'
import ChartContainer from './ChartContainer.vue'
import { ref, provide } from 'vue'
/**
 * @description: 图表容器，用于包裹图表组件，一个图表容器对应一个namespace
 * @file: ChartsContainer.vue
 * @since: 2023-12-09 15:35:28
 **/
const props = defineProps({
  label: {
    type: String,
    required: true
  },
  charts: {
    type: Array,
    required: true
  },
  // namespace的索引
  index: {
    type: Number,
    required: true
  },
  opened: {
    type: Boolean,
    required: true
  },
  isPinned: {
    type: Boolean,
    required: true
  },
  isHidden: {
    type: Boolean,
    required: true
  }
})
const emit = defineEmits(['switch', 'pin', 'unpin', 'hide', 'unhide'])
// ---------------------------------- 控制展开和关闭的状态 ----------------------------------
const isExpand = ref(props.opened)

const handleNamespaceClick = () => {
  isExpand.value = !isExpand.value
  emit('switch', isExpand.value)
}

// ---------------------------------- charts组件列表 ----------------------------------
const chartRefList = ref([])

const setChartRefList = (el, index) => {
  chartRefList.value[index] = el
  chartRefList.value.length = props.charts.length
}

// ---------------------------------- 将charts组建传递给子组件 ----------------------------------
provide('chartRefList', chartRefList)

defineExpose({
  chartRefList
})

// ---------------------------------- 图表置顶、隐藏、取消置顶、取消隐藏 ----------------------------------
const pin = (chart) => {
  emit('pin', chart)
}

const unpin = (chart) => {
  emit('unpin', chart)
}

const hide = (chart) => {
  emit('hide', chart)
}

const unhide = (chart) => {
  emit('unhide', chart)
}
</script>

<style lang="scss" scoped>
.charts-container {
  @apply py-6 px-8;
  &:not(:last-child) {
    @apply border-b;
  }

  .charts-slot {
    @apply grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mt-4.5;
  }
}
</style>
