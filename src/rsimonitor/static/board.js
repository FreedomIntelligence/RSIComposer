"use strict";

const $ = id => document.getElementById(id);
const icon = name => `<i data-lucide="${name}"></i>`;
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const tag = text => `<span class="tag">${escapeHTML(text)}</span>`;
const button = (label, name, action, cls = "") => `<button class="button ${cls}" data-action="${action}">${icon(name)}${label}</button>`;
const sectionTitle = (title, name, right = "", caption = "") => `<div class="section-title"><h2>${icon(name)}${title}${caption ? `<span class="section-caption">${caption}</span>` : ""}</h2><div class="section-actions">${right}</div></div>`;
const table = (headers, rows, cls = "") => `<div class="table-wrap"><table class="${cls}"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.length ? rows.join("") : `<tr><td colspan="${headers.length}" class="empty-state">没有符合条件的记录</td></tr>`}</tbody></table></div>`;
const fact = (label, value) => `<div><div class="fact-label">${label}</div><div class="fact-value">${value}</div></div>`;
const progress = value => `<div class="progress-track"><div class="progress-fill" style="width:${value}%"></div></div>`;
const format = (n, digits = 2) => Number(n).toLocaleString("en-US", {maximumFractionDigits:digits});
const state = {page:"overview", points:true, compareAxis:"round", compareMetric:"dev"};
const labels = {experiments:"实验管理",overview:"运行总览",trajectory:"执行轨迹",metrics:"指标分析",samples:"评测样本",artifacts:"版本与产物",resources:"资源与成本",logs:"日志与配置",compare:"实验对比"};
let chartSpecs = [], chartCounter = 0, toastTimer;
function decorate(){window.lucide?.createIcons();}
function toast(text){$("toast").textContent=text;$("toast").hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").hidden=true,3500);}
function modal(title, body, eyebrow="记录详情"){$("dialog-title").textContent=title;$("dialog-eyebrow").textContent=eyebrow;$("dialog-body").innerHTML=body;$("detail-dialog").showModal();decorate();}
function download(name,text,type="application/json"){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast("记录已导出");}
function stat(label,value,foot,name){return `<div class="stat"><div class="stat-label">${label}${icon(name)}</div><div class="stat-value">${value}</div><div class="stat-foot">${foot}</div></div>`;}
function series(name, color, points) { return {name, color, points}; }
function chart(title,key,value,sets,options={}){
  const id=`chart-${++chartCounter}`;chartSpecs.push({id,title,sets,...options});return `<article class="chart-card ${options.className||""}"><div class="chart-heading"><div><h3>${escapeHTML(title)}</h3><div class="chart-key mono">${escapeHTML(key)}</div></div><div class="chart-top-value">${value}<small>${options.subtitle||""}</small></div></div><div class="chart-wrap"><canvas id="${id}" role="img" aria-label="${escapeHTML(title)}，实验记录曲线" tabindex="0"></canvas><div class="chart-tooltip" hidden></div></div><div class="chart-footer"><div class="legend">${sets.map(s=>`<span><i class="swatch" style="background:${s.color}"></i>${escapeHTML(s.name)}</span>`).join("")}</div><span>${options.axisLabel||"优化器步数"}</span></div></article>`;
}

function render(preserveFocus = false) {
  const focusedId = preserveFocus ? document.activeElement?.id : null;
  chartSpecs = [];
  chartCounter = 0;
  const page = location.hash.slice(1);
  state.page = Object.hasOwn(labels, page) ? page : "overview";
  $("breadcrumb-page").textContent = labels[state.page];
  document.querySelectorAll("[data-nav]").forEach(link => link.classList.toggle("active", link.dataset.nav === state.page));
  $("main").innerHTML = renderLivePage();
  decorate();
  drawCharts();
  if (focusedId) $(focusedId)?.focus({preventScroll: true});
}

function navigate(page) {
  $("detail-dialog").close();
  $("sidebar").classList.remove("open");
  if (location.hash === `#${page}`) render();
  else location.hash = page;
}

document.addEventListener("click", liveClick);
document.addEventListener("change", liveChange);
$("menu-toggle").addEventListener("click",()=>$("sidebar").classList.toggle("open"));
$("dialog-close").addEventListener("click",()=>$("detail-dialog").close());
$("detail-dialog").addEventListener("click",event=>{if(event.target===$("detail-dialog")){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
$("refresh-button").addEventListener("click", refreshLive);
$("run-switch").addEventListener("click", openLiveRuns);
$("workspace-info").addEventListener("click", () => modal("RSISys Research", fact("数据来源", "本地实验的公开快照"), "工作空间"));
window.addEventListener("hashchange", () => {
  $("detail-dialog").close();
  $("sidebar").classList.remove("open");
  render();
  window.scrollTo(0, 0);
  if (state.page === "compare") refreshLive();
});
function drawCharts(){
  const styles=getComputedStyle(document.documentElement),theme={grid:styles.getPropertyValue("--chart-grid").trim(),muted:styles.getPropertyValue("--muted").trim(),paper:styles.getPropertyValue("--paper").trim(),selection:styles.getPropertyValue("--chart-select").trim()};
  for(const spec of chartSpecs){
    const canvas=$(spec.id);if(!canvas)continue;
    const ctx=canvas.getContext("2d"),tooltip=canvas.nextElementSibling;
    let hover=null,zoom=null,drag=null;
    const all=spec.sets.flatMap(s=>s.points),fullMin=Math.min(0,...all.map(p=>p.x)),fullMax=spec.xMax??Math.max(1,...all.map(p=>p.x));
    const bounds=()=>{const r=canvas.getBoundingClientRect();return {w:r.width,h:r.height,left:44,right:15,top:15,bottom:29,xmin:zoom?.[0]??fullMin,xmax:zoom?.[1]??fullMax};};
    function paint(){
      const b=bounds(),dpr=window.devicePixelRatio||1;canvas.width=Math.round(b.w*dpr);canvas.height=Math.round(b.h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,b.w,b.h);ctx.font='9px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      const x=v=>b.left+(v-b.xmin)/(b.xmax-b.xmin||1)*(b.w-b.left-b.right),y=v=>b.h-b.bottom-(v-spec.min)/(spec.max-spec.min)*(b.h-b.top-b.bottom);
      ctx.lineWidth=1;ctx.setLineDash([]);
      for(let i=0;i<5;i++){const v=spec.min+(spec.max-spec.min)*i/4,py=y(v);ctx.strokeStyle=theme.grid;ctx.beginPath();ctx.moveTo(b.left,py);ctx.lineTo(b.w-b.right,py);ctx.stroke();ctx.fillStyle=theme.muted;ctx.textAlign="right";ctx.fillText(`${format(v,Math.abs(spec.max)<3?2:0)}${spec.percent?"%":""}`,b.left-8,py+3);}
      for(let i=0;i<5;i++){const v=b.xmin+(b.xmax-b.xmin)*i/4;ctx.fillStyle=theme.muted;ctx.textAlign="center";ctx.fillText(format(v,Math.abs(fullMax)<10?1:0),x(v),b.h-9);}
      ctx.save();ctx.beginPath();ctx.rect(b.left,b.top,b.w-b.left-b.right,b.h-b.top-b.bottom);ctx.clip();
      for(const set of spec.sets){ctx.strokeStyle=set.color;ctx.lineWidth=1.8;ctx.setLineDash(set.dashed?[4,4]:[]);ctx.beginPath();set.points.forEach((p,i)=>i?ctx.lineTo(x(p.x),y(p.y)):ctx.moveTo(x(p.x),y(p.y)));ctx.stroke();ctx.setLineDash([]);if(state.points||spec.discrete)for(const p of set.points){if(!spec.discrete&&set.points.indexOf(p)%5!==0)continue;ctx.fillStyle=theme.paper;ctx.beginPath();ctx.arc(x(p.x),y(p.y),spec.discrete?3.8:2,0,Math.PI*2);ctx.fill();ctx.stroke();}}
      if(hover){ctx.strokeStyle=theme.muted;ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x(hover.point.x),b.top);ctx.lineTo(x(hover.point.x),b.h-b.bottom);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=hover.set.color;ctx.beginPath();ctx.arc(x(hover.point.x),y(hover.point.y),4,0,Math.PI*2);ctx.fill();ctx.strokeStyle=theme.paper;ctx.stroke();}
      if(drag&&drag.end!==null){ctx.fillStyle=theme.selection;ctx.fillRect(Math.min(drag.start,drag.end),b.top,Math.abs(drag.end-drag.start),b.h-b.top-b.bottom);}
      ctx.restore();
    }
    function inspect(clientX,clientY){const rect=canvas.getBoundingClientRect(),b=bounds(),mx=clientX-rect.left;const xv=b.xmin+(mx-b.left)/(b.w-b.left-b.right)*(b.xmax-b.xmin);let nearest=null;for(const set of spec.sets)for(const point of set.points){if(point.x<b.xmin||point.x>b.xmax)continue;const distance=Math.abs(point.x-xv);if(!nearest||distance<nearest.distance)nearest={set,point,distance};}if(!nearest)return;hover=nearest;tooltip.innerHTML=`<span class="muted">${escapeHTML(nearest.set.name)} · ${format(nearest.point.x,1)}</span><strong>${escapeHTML(spec.title)} ${format(nearest.point.y,3)}${spec.percent?"%":""}</strong>`;tooltip.hidden=false;tooltip.style.left=`${Math.max(4,Math.min(mx-40,rect.width-170))}px`;tooltip.style.top=`${Math.max(3,Math.min(clientY-rect.top-55,rect.height-60))}px`;paint();}
    canvas.addEventListener("pointermove",event=>{if(drag){drag.end=Math.max(bounds().left,Math.min(bounds().w-bounds().right,event.offsetX));paint();}else inspect(event.clientX,event.clientY);});
    canvas.addEventListener("pointerleave",()=>{if(!drag){hover=null;tooltip.hidden=true;paint();}});
    canvas.addEventListener("pointerdown",event=>{if(event.button!==0)return;drag={start:event.offsetX,end:null};canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener("pointerup",event=>{if(!drag)return;const selected=drag;drag=null;if(selected.end!==null&&Math.abs(selected.end-selected.start)>12){const b=bounds(),toValue=p=>b.xmin+(p-b.left)/(b.w-b.left-b.right)*(b.xmax-b.xmin);zoom=[toValue(Math.min(selected.start,selected.end)),toValue(Math.max(selected.start,selected.end))];hover=null;tooltip.hidden=true;paint();}else{inspect(event.clientX,event.clientY);}canvas.releasePointerCapture(event.pointerId);});
    canvas.addEventListener("pointercancel",()=>{drag=null;paint();});canvas.addEventListener("dblclick",()=>{zoom=null;hover=null;tooltip.hidden=true;paint();});
    canvas.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home","End","Escape"].includes(event.key))return;event.preventDefault();if(event.key==="Escape"){hover=null;tooltip.hidden=true;paint();return;}const set=spec.sets[0];let index=hover?set.points.indexOf(hover.point):0;index=event.key==="Home"?0:event.key==="End"?set.points.length-1:Math.max(0,Math.min(set.points.length-1,index+(event.key==="ArrowRight"?1:-1)));hover={set,point:set.points[index]};tooltip.hidden=false;tooltip.style.left="52px";tooltip.style.top="12px";tooltip.innerHTML=`${escapeHTML(set.name)}<strong>${format(hover.point.x)} · ${format(hover.point.y,3)}</strong>`;paint();});
    paint();
  }
}
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>render(true),120);});
render();
startLive();
