"""Domain conversion + durable subprocess + vLLM HTTP fixtures, no real optimization.

Actual Transformers/PEFT and static vLLM proof is a separate operator-run script.
"""
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from examples.medical.native import MedicalTrainingBackend, MedicalServing
from examples.medical.task import samples, encoded, MedicalTask, Benchmark, DataRef
from examples.medical.run import build
from rsisys.adapters.serving import VLLMBackend, VLLMDeploymentBackend
from rsisys.adapters.training import ProcessTrainingBackend, TrainingCommand
from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import ModelRef, BackendBinding, GenerationRequest
from rsisys.loops.contracts.training import TrainingOrigin, ModelActivationPending
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.model import ModelTransportError

BASE = ModelRef('vllm', 'base', 'r0')
TRAINED = replace(BASE, revision='r1')
SOURCE = {'id':'source-1', 'benchmark_id':'medqa', 'question':'Select the square symbol',
          'options':{'A':'square','B':'circle','C':'star','D':'triangle'}, 'gold':'A'}
NATIVE_FIXTURE = r'''
import hashlib, json, os, pathlib, time
root = pathlib.Path(os.environ['RSISYS_JOB_DIR'])
payload = json.loads(pathlib.Path(os.environ['RSISYS_REQUEST_FILE']).read_text())
request = payload['request']
config = json.loads(pathlib.Path(request['recipe_ref']).read_text())
rows = [json.loads(line) for line in pathlib.Path(config['train_dataset']).read_text().splitlines()]
assert config['initial_model'] == request['base']
assert rows and all('input_ids' in row and 'labels' in row for row in rows)
with (root/'starts').open('a') as f: f.write('start\n')
(root/'consumed.json').write_text(json.dumps(rows))
time.sleep(float(os.environ.get('FIXTURE_DELAY','0')))
adapter = root/'adapter'; adapter.mkdir()
evidence = root/'training-evidence.json'
evidence.write_text(json.dumps({'presented_rows':len(rows),'fixture':True}))
result = {'base':request['base'], 'model':dict(request['base'],revision='r1'),
    'candidate':{'candidate_id':root.name,'base_version':request['base']['revision'],
        'parent_system':request['origin']['parent_system'],'affected_roles':['actor'],
        'changes':{'model':{'artifact_id':root.name,'kind':'peft_adapter','uri':adapter.as_uri()},
          'training_evidence':{'artifact_id':'evidence','kind':'training_evidence','uri':evidence.as_uri(),
              'digest':hashlib.sha256(evidence.read_bytes()).hexdigest()}}},
    'metrics':{'loss':1.25},'simulated':True}
pathlib.Path(os.environ['RSISYS_RESULT_FILE']).write_text(json.dumps(result))
'''


class MedicalNativeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = FileArtifactStore(self.root/'artifacts')
        self.source = self.artifacts.put(encoded(SOURCE)+b'\n',kind='training_data')
        self.script = self.root/'native_fixture.py'
        self.script.write_text(NATIVE_FIXTURE)
        self.conversions = []
        def tokenize(rows):
            self.conversions.append(rows)
            return [{'input_ids':[1,2,3], 'labels':[-100,2,3], 'attention_mask':[1,1,1]} for row in rows]
        self.tokenize = tokenize
        self.template = {'model_config':{'model_type':'gpt2','vocab_size':8},
                         'lora_config':{'r':2,'task_type':'CAUSAL_LM'},
                         'training_args':{'use_cpu':True,'max_steps':1}}

    def learner(self, delay='0', **kwargs):
        process = ProcessTrainingBackend(self.root/'process', lambda r,d:
            TrainingCommand((sys.executable,str(self.script)),str(d),{'FIXTURE_DELAY':delay},timeout_seconds=5))
        return MedicalTrainingBackend(self.root/'medical',self.artifacts,tokenize=self.tokenize,
            conversion_id=kwargs.pop('conversion_id','explicit-tokenizer-v1'),
            recipe_template=kwargs.pop('recipe_template',self.template),process_backend=process,**kwargs)

    def request(self, learner, **kwargs):
        return ActorTrainingRequest(BASE,(self.source,),learner.recipe_ref,
            origin=TrainingOrigin('parent-system',BASE),**kwargs)

    def finish(self, learner, job):
        deadline = time.monotonic()+8
        while time.monotonic()<deadline:
            outcome = learner.poll(job)
            if outcome.terminal or outcome.status == JobStatus.UNKNOWN:
                return outcome
            time.sleep(.025)
        self.fail('native fixture did not finish')

    def test_conversion_once_reattach_and_managed_source_and_native_evidence(self):
        learner = self.learner('.1')
        request = self.request(learner)
        before = asdict(request)
        self.assertIs(learner.bind_inputs(request,self.artifacts),request)
        job = learner.submit(request,idempotency_key='same')
        self.assertEqual(learner.submit(request,idempotency_key='same'),job)
        other = self.learner('.1')
        outcome = self.finish(other,job)
        self.assertEqual(outcome.status,JobStatus.SUCCEEDED,outcome.error)
        self.assertEqual(len(self.conversions),1)
        self.assertEqual(asdict(request),before)
        result = other.result(job)
        self.assertEqual(result.base,BASE)
        self.assertEqual(result.model,TRAINED)
        self.assertEqual(result.candidate.parent_system,'parent-system')
        self.assertTrue(result.simulated)
        self.assertEqual(outcome.result,self.learner().reattach(job).result)
        published = {name:json.loads(self.artifacts.get(ref)) for name,ref in result.candidate.changes.items()}
        manifest = published['model']['native_ref']
        self.assertEqual(manifest['kind'],'peft_adapter')
        self.assertTrue(Path(manifest['uri'].removeprefix('file://')).is_dir())
        provenance = published['medical_training']
        self.assertEqual(provenance['request'],json.loads(json.dumps(before)))
        self.assertEqual(provenance['conversion_id'],'explicit-tokenizer-v1')
        self.assertEqual(provenance['sources'][0]['ref']['artifact_id'],self.source.artifact_id)
        self.assertEqual(provenance['source_rows'],1)
        self.assertEqual(provenance['native_rows'],1)
        self.assertIn('receipt.json',provenance['native_job_evidence'])
        self.assertEqual(published['training_evidence']['presented_rows'],1)
        internal = provenance['native_request']
        self.assertEqual(internal['base'],before['base'])
        self.assertEqual(internal['origin'],json.loads(json.dumps(before['origin'])))
        self.assertEqual(internal['data_refs'][0]['uri'].split(':')[0],'file')
        native_dir = Path(provenance['native_job_directory'])
        self.assertEqual((native_dir/'starts').read_text(),'start\n')
        self.assertEqual(json.loads((native_dir/'consumed.json').read_text())[0]['labels'],[-100,2,3])

    def test_unknown_recipe_and_conflicting_retries_rejected(self):
        learner = self.learner()
        request = self.request(learner)
        with self.assertRaisesRegex(ValueError,'recipe_ref'):
            learner.submit(replace(request,recipe_ref='invented'),idempotency_key='bad')
        self.assertEqual(self.conversions,[])
        job = learner.submit(request,idempotency_key='fixed')
        self.assertEqual(self.finish(learner,job).status,JobStatus.SUCCEEDED)
        changed = self.artifacts.put(encoded({**SOURCE,'id':'source-2'})+b'\n',kind='training_data')
        with self.assertRaisesRegex(ValueError,'different request'):
            learner.submit(replace(request,data_refs=(changed,)),idempotency_key='fixed')
        with self.assertRaisesRegex(ValueError,'original conversion'):
            self.learner(conversion_id='changed').reattach(job)
        with self.assertRaisesRegex(ValueError,'original conversion'):
            self.learner(recipe_template={**self.template,'seed':99}).reattach(job)

    def test_template_binding_features_and_duplicate_sources_fail_before_job(self):
        bad = self.learner(recipe_template={**self.template,'initial_model':asdict(TRAINED)})
        with self.assertRaisesRegex(ValueError,'initial_model'):
            bad.submit(self.request(bad),idempotency_key='base')
        learner = self.learner()
        learner.tokenize = lambda rows: [{'text':'implicit tokenization is forbidden'}]
        with self.assertRaisesRegex(ValueError,'feature dictionaries'):
            learner.submit(self.request(learner),idempotency_key='features')
        with self.assertRaisesRegex(ValueError,'unique'):
            learner.submit(replace(self.request(learner),data_refs=(self.source,self.source)),idempotency_key='duplicates')
        self.assertEqual(list((self.root/'process').glob('*/receipt.json')),[])

    def test_cancel_after_reattach_and_result_does_not_advance(self):
        learner = self.learner('20')
        job = learner.submit(self.request(learner),idempotency_key='cancel')
        with self.assertRaises(RuntimeError): learner.result(job)
        self.learner().cancel(job)
        self.assertEqual(self.finish(learner,job).status,JobStatus.CANCELLED)
        self.assertEqual(len(self.conversions),1)

    def test_build_binding_accepts_the_domain_wrapper(self):
        learner = self.learner()
        train = self.root/'train.jsonl'; train.write_bytes(encoded(SOURCE)+b'\n')
        dev = self.root/'dev.jsonl'; dev.write_bytes(encoded({**SOURCE,'id':'dev','question':'Select the circle symbol'})+b'\n')
        task = MedicalTask(DataRef.freeze(train),(Benchmark('medqa','primary',DataRef.freeze(dev)),))
        from rsisys.runtime.systems import ReferenceSystemActivator
        # Construction only: no native job or inference is started here.
        assembly = build(task,artifacts=self.artifacts,decision_backend=object(),decision_model=BASE,
            initial_model=BASE,inference=lambda request:None,system_activator=ReferenceSystemActivator(BASE),
            learner=learner,bind_training_inputs=learner.bind_inputs)
        self.assertIs(assembly.runner.exploration.harness.learner,learner)
        self.assertEqual(self.conversions,[])


class VLLMHTTPFixture:
    def __init__(self):
        self.rows = {'base':{'id':'base','root':'base'}}
        self.requests=[]
        self.pending=False
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def reply(self,value,status=200,plain=False):
                self.send_response(status); self.end_headers()
                self.wfile.write(value.encode() if plain else encoded(value))
            def do_GET(self):
                owner.requests.append(('GET',self.path,None))
                self.reply({'data':list(owner.rows.values())})
            def do_POST(self):
                value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append(('POST',self.path,value))
                if self.path.endswith('/load_lora_adapter'):
                    owner.rows[value['lora_name']]={'id':value['lora_name'],'root':value['lora_path']}
                    self.reply('Success',plain=True)
                elif self.path.endswith('/unload_lora_adapter'):
                    owner.rows.pop(value['lora_name'],None); self.reply('Success',plain=True)
                elif value['model'] not in owner.rows:
                    self.reply({'error':'not found'},404)
                elif owner.pending:
                    self.reply({'error':'not ready'},503)
                else:
                    self.reply({'model':value['model'],'choices':[{'finish_reason':'stop',
                        'message':{'content':'A'}}]})
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


class MedicalServingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture=VLLMHTTPFixture(); self.addCleanup(self.fixture.close)
        self.binding=BackendBinding('vllm','http',f'http://127.0.0.1:{self.fixture.server.server_port}/v1',json_mode=False)
        self.backend=VLLMBackend(self.binding)
    def deployments(self,**kwargs):
        return VLLMDeploymentBackend(self.backend,Path(self.temp.name)/'deployments',**kwargs)
    def test_first_generate_pins_and_new_revision_cannot_fallback(self):
        self.assertFalse(self.backend.generate(GenerationRequest('actor',BASE,'Q')).revision_verified)
        before=len(self.fixture.requests)
        with self.assertRaises(ValueError): self.backend.generate(GenerationRequest('actor',TRAINED,'Q'))
        with self.assertRaises(ValueError): self.backend.probe(TRAINED)
        self.assertEqual(len(self.fixture.requests),before)
        self.backend.bind(TRAINED,'isolated')
        self.fixture.rows['isolated']={'id':'isolated','root':'owned'}
        self.backend.generate(GenerationRequest('actor',TRAINED,'Q'))
        self.assertEqual(self.fixture.requests[-1][2]['model'],'isolated')
    def test_first_probe_pins_before_generate_and_concurrent_revisions(self):
        self.backend.probe(BASE)
        before=len(self.fixture.requests)
        with self.assertRaises(ValueError): self.backend.generate(GenerationRequest('actor',TRAINED,'Q'))
        self.assertEqual(len(self.fixture.requests),before)
        from concurrent.futures import ThreadPoolExecutor
        backend=VLLMBackend(self.binding)
        def generate(model):
            try: backend.generate(GenerationRequest('actor',model,'Q')); return 'ok'
            except ValueError: return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(generate,(BASE,TRAINED))),['ok','conflict'])
    def test_static_preparation_and_activation_are_read_only(self):
        serving=MedicalServing(self.deployments())
        system=SystemVersion('initial',{},BASE)
        self.assertEqual(serving.prepare_inference(system),system)
        self.assertEqual(serving.activate(system,idempotency_key='apply'),system)
        self.assertEqual(serving(GenerationRequest('actor',BASE,'Q')).model,BASE)
        self.assertFalse(any('lora_adapter' in path for _,path,_ in self.fixture.requests))
        with self.assertRaisesRegex(ValueError,'different system'):
            serving.activate(replace(system,version='other'),idempotency_key='apply')
    def test_candidate_prepared_before_inference_dynamic_path_and_no_duplicate_load(self):
        deployments=self.deployments(allow_runtime_lora=True)
        seen=[]
        def path(system):
            seen.append(system.version)
            return None if system.model==BASE else '/owned/native-adapter'
        serving=MedicalServing(deployments,adapter_path=path)
        base=SystemVersion('initial',{},BASE)
        serving.prepare_inference(base)
        candidate=SystemVersion('candidate',{},TRAINED,'initial')
        self.assertEqual(serving.prepare_inference(candidate),candidate)
        self.assertEqual(serving.generate(GenerationRequest('actor',TRAINED,'Q')).model,TRAINED)
        alias=self.fixture.requests[-1][2]['model']
        self.assertNotEqual(alias,'base')
        self.assertEqual(serving.activate(candidate,idempotency_key='apply-candidate'),candidate)
        serving.prepare_inference(replace(candidate,version='prompt-variant',parent_version='candidate'))
        self.assertEqual(sum(path.endswith('/load_lora_adapter') for _,path,_ in self.fixture.requests),1)
        self.assertEqual(deployments.adapter_paths[TRAINED],'/owned/native-adapter')
        serving.adapter_path=lambda system:'/different/weights'
        with self.assertRaisesRegex(ValueError,'different native adapter'):
            serving.prepare_inference(candidate)
        self.assertEqual(sum(path.endswith('/load_lora_adapter') for _,path,_ in self.fixture.requests),1)
    def test_runtime_opt_in_and_pending_preparation(self):
        serving=MedicalServing(self.deployments(),adapter_paths={TRAINED:'/owned/adapter'})
        with self.assertRaises(NotImplementedError): serving.prepare_inference(SystemVersion('candidate',{},TRAINED))
        self.fixture.pending=True
        serving=MedicalServing(self.deployments())
        with self.assertRaises(ModelActivationPending): serving.prepare_inference(SystemVersion('initial',{},BASE))
        self.fixture.pending=False
        self.assertEqual(serving.prepare_inference(SystemVersion('initial',{},BASE)).model,BASE)
    def test_unload_never_restores_base_fallback_even_without_baseline_pin(self):
        deployments=self.deployments(adapter_paths={TRAINED:'/owned/adapter'},allow_runtime_lora=True)
        state=deployments.prepare(TRAINED,idempotency_key='candidate-only')
        alias=self.backend.models[TRAINED]
        self.assertNotEqual(alias,BASE.model_id)
        deployments.release(state.deployment,idempotency_key='release')
        with self.assertRaises(ModelTransportError): self.backend.generate(GenerationRequest('actor',TRAINED,'Q'))
        self.assertEqual(self.fixture.requests[-1][2]['model'],alias)
        before=len(self.fixture.requests)
        with self.assertRaises(ValueError): self.backend.probe(replace(TRAINED,revision='different'),alias)
        self.assertEqual(len(self.fixture.requests),before)


if __name__=='__main__': unittest.main()
