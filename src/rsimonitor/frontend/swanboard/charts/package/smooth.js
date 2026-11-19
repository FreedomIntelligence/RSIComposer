/*
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: validated per-chart settings, unit-independent EMA
coordinates, and a Gaussian sampling radius based on the standard deviation.
See the repository LICENSE for the complete Apache-2.0 terms.
*/
/**
 * 数据平滑处理
 */

// 时间加权均衡
const timeEMA = (data, param) => {
  // 基础smoothingWeight
  const smoothingWeight = Math.min(Math.sqrt(param || 0), 0.999)
  let debiasWeight = 0
  const rangeOfX = data.length > 1 ? data[data.length - 1].index - data[0].index : 0
  // 上一个点的y值
  let lastY = data.length > 0 ? 0 : NaN
  return data.map((item, index) => {
    const prevX = index > 0 ? index - 1 : 0
    // VIEWPORT_SCALE scales the result to the chart's x-axis range
    const changeInX = rangeOfX > 0 ? ((data[index].index - data[prevX].index) / rangeOfX) * 1000 : 0
    const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX)
    lastY = lastY * smoothingWeightAdj + data[index].data
    debiasWeight = debiasWeight * smoothingWeightAdj + 1
    return { ...item, data: lastY / debiasWeight }
  })
}

// 流动平均
const runningAverage = (data, param) => {
  // 理论上的窗口大小
  const w = Math.floor(param / 2)
  let floorX, ceilX, nowIndex
  let sum = 0
  return data.map((item, index) => {
    floorX = index - w
    ceilX = index + w
    sum = 0
    for (let i = floorX; i <= ceilX; i++) {
      nowIndex = Math.min(Math.max(i, 0), data.length - 1)
      sum += data[nowIndex].data
    }
    return { ...item, data: sum / (ceilX - floorX + 1) }
  })
}

// 创建高斯函数
const createGaussian = () => {
  // 根号2π
  const sqrt2PI = Math.sqrt(2 * Math.PI)
  /**
   * 高斯函数
   * @param { Number } x 自变量
   * @param { Number } stdDev 标准差
   */
  return (x, stdDev) => {
    return Math.exp(-0.5 * Math.pow(x / stdDev, 2)) / (stdDev * sqrt2PI)
  }
}

// 高斯平均
const gaussianAverage = (data, param) => {
  const gaussian = createGaussian()
  // 在代码中正负无穷似乎在大数据上影响算法速度，但是高斯函数在-3到3区间上进行积分似乎已经达到0.99以上，考虑使用此区间即可
  const range = Math.min(data.length - 1, Math.ceil(3 * param))
  // 对于每一个值，计算高斯函数的值，只采用-3到3的区间
  let sum = 0
  let weightSum = 0
  return data.map((item, index) => {
    sum = 0
    weightSum = 0
    for (let i = -range; i <= range; i++) {
      if (index + i >= 0 && index + i < data.length) {
        sum += data[index + i].data * gaussian(i, param)
        weightSum += gaussian(i, param)
      }
    }
    return { ...item, data: sum / weightSum }
  })
}

// 平滑算法映射关系
const smoothArithmetics = new Map([
  [0, undefined],
  [1, timeEMA],
  [2, runningAverage],
  [3, gaussianAverage]
])

/**
 * 数据平滑处理，传入数据和平滑方法
 * @param { Array } data 数据
 * @param { Object } method 平滑方法
 * @param { Number } method.id 平滑方法id
 * @param { Number} method.value 平滑方法参数
 * @returns { Array } 平滑后的数据
 */
export default function smooth(data, method) {
  if (!needSmooth(method)) return data.map(item => ({...item}))
  return smoothArithmetics.get(method.id)(data, method.value)
}

/**
 * 判断是否需要平滑，传入平滑方法
 * @param { Object } method 平滑方法
 * @param { Number } method.id 平滑方法id
 * @param { Number} method.value 平滑方法参数
 * @returns { Boolean } 是否需要平滑，true为需要，false为不需要
 */
export const needSmooth = method => !!method && [1, 2, 3].includes(method.id) && Number.isFinite(method.value) && method.value > 0

// Panel settings use the same algorithm IDs as the ported SwanBoard renderer.
export const SMOOTHING_METHODS = [
  {id:0,name:'不平滑',defaultValue:0,description:'展示原始指标曲线。'},
  {id:1,name:'Time Weighted EMA',min:0,max:0.99,step:0.01,defaultValue:0.6,parameter:'平滑系数',description:'时间加权指数移动平均，系数越大，历史数据的权重越高。'},
  {id:2,name:'Running Average',min:1,max:100,step:1,defaultValue:5,parameter:'窗口大小',description:'使用相邻数据点的平均值查看趋势，窗口越大，曲线越平滑。'},
  {id:3,name:'Gaussian Smoothing',min:1,max:100,step:1,defaultValue:3,parameter:'标准差',description:'使用高斯权重对邻近数据点进行平滑。'}
]
export function normalizeSmoothing(value) {
  if (value === undefined || value === null) return {id:0,value:0,showOriginal:true}
  if (typeof value !== 'object' || Array.isArray(value)) throw new Error('平滑配置无效')
  const method = SMOOTHING_METHODS.find(method => method.id === value.id)
  if (!method) throw new Error('请选择有效的平滑算法')
  if (value.showOriginal !== undefined && typeof value.showOriginal !== 'boolean') throw new Error('原始曲线配置无效')
  const result = {id:method.id,value:0,showOriginal:value.showOriginal !== false}
  if (method.id === 0) return result
  if (!['number','string'].includes(typeof value.value) || value.value === '') throw new Error('请填写平滑参数')
  const number = Number(value.value)
  if (!Number.isFinite(number) || number < method.min || number > method.max || method.step === 1 && !Number.isInteger(number)) throw new Error(`${method.name} 参数范围为 ${method.min}–${method.max}${method.step === 1 ? '（整数）' : ''}`)
  result.value = number
  return result
}
