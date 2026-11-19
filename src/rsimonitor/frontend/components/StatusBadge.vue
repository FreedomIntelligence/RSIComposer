<template><span class="badge" :class="tone"><i></i>{{ t(LABELS[status] || status || '未上报') }}</span></template>
<script setup>
import {computed} from 'vue';
import {t} from '../app/i18n.js';
const props = defineProps({status: {type: String, default: ''}});
const LABELS = {
  unavailable: '快照不可用', starting: '启动中', queued: '排队中', pending: '等待中', running: '运行中', active: '运行中',
  executing: '运行中', waiting: '等待中', progressed: '进行中', completed: '已完成', finished: '已完成', succeeded: '已完成',
  failed: '失败', cancelled: '已取消', stopped: '已停止', paused: '已暂停', partial: '部分完成', step_limit: '达到步数上限',
  proposed: '已提出', reviewed: '已复查', cancel_requested: '已请求取消',
};
const tone = computed(() => ['completed', 'finished', 'succeeded'].includes(props.status) ? 'ok'
  : ['failed', 'unavailable'].includes(props.status) ? 'bad'
  : ['running', 'active', 'executing', 'progressed', 'starting'].includes(props.status) ? 'live' : 'idle');
</script>
