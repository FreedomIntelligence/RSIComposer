/*
Adapted from SwanLab-Dashboard, revision e2a7a84f49dcf18ff09c50846f5516b69093c0b9.
Source: https://github.com/SwanHubX/SwanLab-Dashboard
Copyright SwanLab-Dashboard contributors. SPDX-License-Identifier: Apache-2.0
Modified for RSIMonitor: local data binding, scoped styling and host navigation.
See the repository LICENSE for the complete Apache-2.0 terms.
*/
import zhCN from './zh-CN'
import enUS from './en-US'
import { createI18n } from 'vue-i18n'

export const langs = ['zh-CN', 'en-US']

export const getDefaultLang = () => {
  // 从localStorage中获取语言设置
  const lang = localStorage.getItem('lang')
  if (lang && langs.includes(lang)) {
    return lang
  }
  // 或者从浏览器中获取语言设置
  const navLang = navigator.language
  if (langs.includes(navLang)) {
    return navLang
  }
  // 默认返回英文
  return 'en-US'
}

// 设置html标签的lang属性
// Host document language belongs to RSIMonitor.

export const i18n = createI18n({
  // locale: 'en', // 设置地区
  locale: document.documentElement.lang.startsWith('zh') ? 'zh-CN' : getDefaultLang(),
  legacy: false, // 如果要支持compositionAPI，此项必须设置为false
  globalInjection: true, // 全局注册$t方法
  messages: {
    'zh-CN': zhCN,
    'en-US': enUS
  }
})

// 导出国际化实例
export default i18n
// 在js中使用国际化
export const { t, te, tm } = i18n.global
