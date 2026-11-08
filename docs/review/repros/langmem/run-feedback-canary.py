from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
from rsisys.adapters.serving import VLLMBackend
from rsisys.loops.contracts.model_service import BackendBinding,ModelRef
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.receipts import ExecutionReceipt
from rsisys.loops.decision_making.feedback import FeedbackEvidence
from rsisys.loops.decision_making.feedback.langmem import LangMemContextMemory
from rsisys.loops.decision_making.feedback.learning import FeedbackLearning,SerializedLearningBackend
from rsisys.loops.decision_making.feedback.service import FeedbackLearningService
from rsisys.runtime.files import atomic_write,json_bytes
from rsisys.runtime.json_tools import result_id
from tests.feedback.test_feedback_candidate_evidence import CALL48

root=Path('.local/validation/langmem')/('feedback-canary-'+str(time.time_ns()))
root.mkdir()
model=ModelRef('meta','Qwen3.8-27B','shared-service-label')
class ObservedBackend(VLLMBackend):
 def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs);self.calls=0
 def generate(self,request):
  self.calls+=1
  if self.calls>24: raise RuntimeError('Canary host model call cap reached')
  return self.generate_with_response_observer(request,lambda wire: atomic_write(root/f'wire-{self.calls}.json',json_bytes(wire)))
rawbackend=ObservedBackend(BackendBinding('meta','chat_completions',endpoint='http://127.0.0.1:18880/v1',timeout_seconds=90),generation_options={'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
backend=SerializedLearningBackend(rawbackend)
provider=LangMemContextMemory(root/'memory.sqlite',namespace=('validation','feedback'))
learning=FeedbackLearning(backend,path=root/'feedback.json',processor=provider,context_memory=provider,max_calls=24,max_input_bytes=24000)
service=FeedbackLearningService(learning,SimpleNamespace(model=lambda:model,publish=lambda m:None))
@dataclass
class Eval:
 candidate_id:str
 directive_id:str
 protocol_ref:str
 status:str
 quality_metrics:dict
 simulated:bool=True
 input_refs:tuple=()
 evidence_refs:tuple=()
evalA=Eval('candidate-A','eval-A','dev-v1','completed',{'accuracy':0.25})
results=[]
def finish(ident):
 until=time.monotonic()+180
 while learning.state['jobs'][ident]['status'] in ('queued','running') and time.monotonic()<until:
  service.refresh();time.sleep(0.1)
 row=learning.state['jobs'][ident]
 entry={'job_id':ident,'status':row['status'],'error':row['error'],'charged_calls':row.get('charged_calls'),'view':service.working_memory()}
 results.append(entry)
 atomic_write(root/'report.json',json_bytes({'scope':'Real inference over synthetic feedback and a historical public receipt; no training or research run','configuration':provider.configuration(),'host_calls':rawbackend.calls,'attempted':learning.state['attempted'],'results':results}))
 print(json.dumps(entry,ensure_ascii=False),flush=True)
 return row['status']=='completed'
try:
 for i in range(2):
  d=Directive('train-B-'+str(i),'Inspect synthetic memory behavior','The same B training observation is repeated; no new evaluation.',mode='train',simulated=True)
  f=Feedback(d.directive_id,ExecutionStatus.COMPLETED,'Candidate B training completed; B has not been evaluated or adopted. Candidate A evaluation is supplied separately.',True,candidates=(Candidate('candidate-B','initializer',{},('actor',)),))
  ident=service.observe(ExecutionReceipt.from_feedback(d,f),evidence=FeedbackEvidence(evaluations=(evalA,)))
  if not finish(ident):break
 else:
  d=Directive('eval-A-correction','Inspect synthetic memory correction','Correct the earlier reported A accuracy using this exact evaluation record. B remains unevaluated.',mode='evaluate',simulated=True,payload={'candidate_id':'candidate-A','protocol_ref':'dev-v1'})
  f=Feedback(d.directive_id,ExecutionStatus.COMPLETED,'Correction: A accuracy is 0.21875; previous A accuracy=0.25 was a reporting error. B remains unevaluated and unadopted.',True,quality_metrics={'accuracy':0.21875})
  finish(service.observe(ExecutionReceipt.from_feedback(d,f)))
 # Replay a pinned, already-public historical bug receipt, not protected data.
 raw=deepcopy(CALL48['raw'])
 row={'job_id':'historical-call48','capture_index':len(learning.state['jobs']),'receipt_digest':result_id(raw['receipt']),'model':asdict(model),'raw_input':raw,'status':'queued','proposals':[],'error':None,'simulated':False,'response':None}
 assert learning._bound_input(row)
 learning._commit({**learning.state,'jobs':{**learning.state['jobs'],row['job_id']:row}})
 finish(row['job_id'])
finally:
 service.close(timeout=100)
print(root,flush=True)
