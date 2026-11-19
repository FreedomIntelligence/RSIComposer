import fs from 'node:fs';
import {notices} from './upstream-notices.js';
import path from 'node:path';
export function attributions(){
  return {name:'rsimonitor-third-party-notices',generateBundle(_options,bundle){
    const roots=new Set();
    for(const output of Object.values(bundle))if(output.type==='chunk')for(const id of Object.keys(output.modules)){
      const marker='/node_modules/',index=id.lastIndexOf(marker);if(index<0)continue;
      const remaining=id.slice(index+marker.length).split('/');
      const name=remaining[0].startsWith('@')?remaining.slice(0,2).join('/'):remaining[0];
      roots.add((id.slice(0,index+marker.length)+name).replaceAll('\0',''));
    }
    let notice='RSIMonitor chart workspace\nAdapted from SwanHubX/SwanLab-Dashboard\nRevision e2a7a84f49dcf18ff09c50846f5516b69093c0b9\nCopyright SwanLab-Dashboard contributors. Apache-2.0.\nModified for RSISys: local data subscriptions, host UI, persistent layouts and axes.\n';
    notice+=fs.readFileSync(path.resolve('../../../LICENSE'),'utf8')+'\n\nBundled dependencies:\n';
    for(const root of [...roots].sort()){
      const metadata=JSON.parse(fs.readFileSync(path.join(root,'package.json'),'utf8'));
      const files=fs.readdirSync(root).filter(name=>/^(license|licence|copying|notice)(\.|$|-)/i.test(name));
      notice+=`\n${metadata.name} ${metadata.version}\n`;
      if(!files.length){
        const declared=metadata.license;
        if(!['MIT','ISC'].includes(declared))throw new Error('Unknown bundled license: '+metadata.name);
        notice+='Declared license: '+declared+'\nUpstream author metadata: '+JSON.stringify(metadata.author||metadata.repository)+'\n';
        const terms=notices[declared];
        notice+='SPDX terms: '+terms.source+'\n'+terms.text.replace(/^Copyright.*\n/m,'')+'\n';
        const project=metadata.name.startsWith('@antv/')?'AntV':metadata.name==='@headlessui/vue'?'HeadlessUI':metadata.name==='@vue/devtools-api'?'VueDevtools':null;
        if(project)notice+=notices[project].source+'\n'+notices[project].text+'\n';
      }
      for(const name of files)notice+=fs.readFileSync(path.join(root,name),'utf8')+'\n';
    }
    for(const output of Object.values(bundle))if(output.type==='chunk')output.code='/*!\n'+notice.replaceAll('*/','* /')+'\n*/\n'+output.code;
  }};
}
