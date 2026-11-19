/*
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
*/
/**
 * 在此处封装一些图表可共用的函数逻辑
 */

/**
 * 将refrence字段映射为x轴字段和显示的x轴名称
 */
export const refrence2XField = {
  step: { xField: 'index', xTitle: 'Step' },
  elapsed_seconds: { xField: 'index', xTitle: 'Elapsed / s' },
  observation: { xField: 'index', xTitle: '' },
  timestamp: { xField: 'index', xTitle: 'Time' },
  metric: { xField: 'index', xTitle: 'Metric' }
}

export const transparentColor = (color, opacity = 0.1) => {
  // 将十六进制增加透明度通道
  opacity = Math.floor(opacity * 255)
  // opacity转为十六进制
  const opacityHex = opacity.toString(16)
  return color + opacityHex
}
