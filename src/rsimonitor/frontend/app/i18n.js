/** UI copy is keyed by its Chinese source text; reported model prose never passes through t(). */
import {ref} from 'vue';
import {english} from './messages.js';

const STORAGE_KEY = 'rsicomposer-language';
export const language = ref('zh');
try {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === 'zh' || saved === 'en') language.value = saved;
} catch { /* Storage is optional. */ }

export function t(key, params = {}) {
  if (key === undefined || key === null) return '';
  const text = language.value === 'en' && Object.hasOwn(english, key) ? english[key] : String(key);
  return text.replace(/\{(\w+)\}/g, (token, name) => Object.hasOwn(params, name) ? String(params[name]) : token);
}

export function setLanguage(next) {
  if (next !== 'zh' && next !== 'en') return;
  language.value = next;
  document.documentElement.lang = next === 'zh' ? 'zh-CN' : 'en';
  try { localStorage.setItem(STORAGE_KEY, next); } catch { /* Keep the in-memory preference. */ }
}

export function dateText(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(language.value === 'zh' ? 'zh-CN' : 'en-US', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).format(date);
}

export function timeText(value) {
  const date = new Date(value);
  if (!value || Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('en-GB', {hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).format(date);
}

export function durationText(seconds) {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return t('{p0} 秒', {p0: Number(seconds.toFixed(seconds < 10 ? 2 : 0))});
  const whole = Math.floor(seconds), hours = Math.floor(whole / 3600), minutes = Math.floor(whole % 3600 / 60);
  return [hours ? t('{p0} 小时', {p0: hours}) : '', t('{p0} 分', {p0: minutes}), t('{p0} 秒', {p0: whole % 60})]
    .filter(Boolean).join(' ');
}
