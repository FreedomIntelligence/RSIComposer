import json, time
from pathlib import Path
from langmem import create_memory_store_manager
from langgraph.store.sqlite import SqliteStore
from rsisys.adapters.langchain_model import BackendChatModel
from rsisys.adapters.serving import VLLMBackend
from rsisys.loops.contracts.model_service import BackendBinding,ModelRef
from rsisys.loops.decision_making.feedback.learning import SerializedLearningBackend
from rsisys.loops.decision_making.feedback.model_gateway import FeedbackModelGateway
root=Path('.local/validation/langmem') / ('native-default-'+str(time.time_ns()))
root.mkdir()
model=ModelRef('meta','Qwen3.8-27B','shared-service-label')
backend=SerializedLearningBackend(VLLMBackend(BackendBinding('meta','chat_completions',endpoint='http://127.0.0.1:18880/v1',timeout_seconds=90), generation_options={'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}))
results=[]
with SqliteStore.from_conn_string(str(root/'memory.sqlite')) as store:
 store.setup()
 for i,text in enumerate([
  'Research feedback: candidate A was evaluated on protocol dev-v1, accuracy=0.25. Candidate B has only been trained, not evaluated or adopted. Do not attribute A metrics to B.',
  'Research feedback: candidate A was evaluated on protocol dev-v1, accuracy=0.25. Candidate B has only been trained, not evaluated or adopted. This repeats the same observation.',
  'Correction: candidate A accuracy on protocol dev-v1 was misreported. The exact evaluation record states accuracy=0.21875, not 0.25. Candidate B is still unevaluated and unadopted.'
 ]):
  gateway=FeedbackModelGateway(backend,model,max_calls=6,max_input_tokens=16000,path=root/f'exchanges-{i}.json')
  chat=BackendChatModel(backend=gateway,model_ref=model,max_output_tokens=2048,cache=False)
  manager=create_memory_store_manager(chat,namespace=('native','default'),store=store,enable_deletes=True)
  try:
   puts=manager.invoke({'messages':[{'role':'user','content':text}],'max_steps':1},config={'recursion_limit':12,'max_concurrency':1})
   items=[x.dict() for x in store.search(('native','default'),limit=20)]
   results.append({'input':text,'puts':puts,'calls':gateway.attempted,'memories':items})
  except Exception as e:
   results.append({'error':f'{type(e).__name__}: {e}','calls':gateway.attempted});break
(root/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str))
print(root)
print(json.dumps(results,ensure_ascii=False,default=str))
