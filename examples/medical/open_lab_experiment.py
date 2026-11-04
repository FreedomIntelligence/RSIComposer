"""Real medical meta-agent over the replaceable Open Lab assembly.

This launcher owns time, admission, evidence and device routing. Algorithms stay
in model-written programs; native Lab outputs are observations, not validated
candidate adoption. Use only in a trusted, scheduler-isolated allocation.
"""
import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import traceback

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.adapters.serving import VLLMBackend
from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import ModuleAssembly, build_default_decision
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.policy.prompts import DecisionPrompts
from rsisys.research import ResearchAssembly
from rsisys.runtime.files import atomic_write, json_bytes
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.model import ModelResponseError
from rsisys.runtime.run_policy import TimeRunPolicy, RunDecision


def write(path, value):
    atomic_write(Path(path), json_bytes(value))


def append(path, value):
    with Path(path).open('ab') as stream:
        stream.write(json_bytes(value) + b'\n')


class HostAdmissionStop(RuntimeError):
    pass


class AbsoluteHostPolicy:
    def __init__(self, backend, seconds, deadline_epoch=None):
        self.backend, self.timed, self.deadline_epoch = backend, TimeRunPolicy(seconds), deadline_epoch

    def evaluate(self, context):
        if self.backend.stop_requested or (self.deadline_epoch is not None and time.time() >= self.deadline_epoch):
            return RunDecision('stop', 'absolute_host_deadline_or_signal', 'cancel')
        return self.timed.evaluate(context)


def sglang_template_tools(request):
    """Mirror the deployed SGLang 0.5.20 Tool.model_dump() before rendering.

    The server adds function.strict=False and tool.defer_loading=None; counting
    the raw wire declarations undercounts their rendered schema. This belongs
    to this deployment, rather than every interchangeable model backend.
    """
    if request.tool_choice == 'none':
        return []
    tools = request.chat_tools().get('tools', [])
    if isinstance(request.tool_choice, dict):
        name = request.tool_choice['function']['name']
        tools = [tool for tool in tools if tool['function']['name'] == name]
    normalized = []
    for tool in tools:
        raw = tool['function']
        function = {'description': raw.get('description'), 'name': raw['name'],
                    'parameters': raw.get('parameters'), 'strict': raw.get('strict', False)}
        deferred = raw.get('defer_loading')
        if deferred is None:
            deferred = tool.get('defer_loading')
        if deferred is not None:
            function['defer_loading'] = deferred
        normalized.append({'type': tool['type'], 'function': function,
                           'defer_loading': tool.get('defer_loading')})
    return normalized


class RecordedBackend:
    """One account for decisions, feedback, summaries and rejected responses."""
    supports_tool_calling = True
    supports_response_schema = True
    token_count_mode = 'local_official_chat_template'

    def __init__(self, transport, tokenizer, root, *, max_calls, max_input_tokens=56000,
                 template_tool_normalizer=None):
        self.transport, self.tokenizer, self.root = transport, tokenizer, Path(root)
        self.template_tool_normalizer = template_tool_normalizer
        self.max_calls, self.max_input_tokens = max_calls, max_input_tokens
        self.calls, self.deadline = 0, None
        self.stop_requested = False
        self.lock = threading.RLock()

    def count_tokens(self, request):
        tools = (self.template_tool_normalizer(request) if self.template_tool_normalizer
                 else list(request.tools))
        kwargs = {'tools': tools} if tools else {}
        tokens = self.tokenizer.apply_chat_template(request.chat_messages(), tokenize=True,
            add_generation_prompt=True, enable_thinking=False, **kwargs)
        # Transformers 5 returns BatchEncoding (a Mapping, not a dict). Its
        # length counts fields, usually 2, rather than prompt tokens.
        return len(tokens['input_ids'] if isinstance(tokens, Mapping) else tokens)

    def generate(self, request):
        with self.lock:
            original = request
            for attempt in range(3):
                try:
                    return self._generate(request, response_attempt=attempt + 1)
                except ModelResponseError as exc:
                    # Native memory tools own their validation/recovery lifecycle.
                    # Only retry rejected decision JSON, before any action exists.
                    if (not exc.retryable or attempt == 2 or original.tools
                            or original.role != 'decision_maker' or original.response_schema is None):
                        raise
                    correction = ('The previous response exceeded the output limit. Return a shorter, complete JSON object.'
                                  if exc.reason == 'output_truncated' else
                                  'The previous response failed format validation. Return one complete valid JSON object.')
                    request = replace(original, system=original.system + '\n' + correction +
                        ' Preserve the required schema and evidence; do not execute or assume any previous action.')
            raise AssertionError('Unreachable response recovery loop')

    def _generate(self, request, *, response_attempt):
        with self.lock:
            if self.stop_requested or self.calls >= self.max_calls or (self.deadline is not None and time.monotonic() >= self.deadline):
                raise HostAdmissionStop('Host inference admission closed')
            count = self.count_tokens(request)
            if count > self.max_input_tokens:
                raise ValueError('Complete request exceeds configured model input limit')
            self.calls += 1
            ident = self.calls
            append(self.root / 'model_requests.jsonl', {'call': ident, 'at': time.time(),
                'response_attempt': response_attempt,
                'input_tokens': count, 'request': asdict(request),
                'wire_request': self.transport._generation_payload(request)})
            write(self.root / 'model_calls.json', {'attempted': self.calls, 'limit': self.max_calls})
            try:
                response = self.transport.generate_with_response_observer(request,
                    lambda wire: append(self.root / 'model_wire.jsonl', {
                        'call': ident, 'response_attempt': response_attempt, **wire}))
                append(self.root / 'model_responses.jsonl', {'call': ident, 'at': time.time(),
                    'response': asdict(response)})
                return response
            except BaseException as exc:
                append(self.root / 'model_errors.jsonl', {'call': ident, 'at': time.time(),
                    'response_attempt': response_attempt,
                    **({'reason': exc.reason, 'retryable': exc.retryable,
                        'finish_reason': exc.finish_reason, 'provider_model': exc.provider_model}
                       if isinstance(exc, ModelResponseError) else {}),
                    'error': f'{type(exc).__name__}: {exc}'})
                raise


class TimedOpenPolicy(ModelDecisionPolicy):
    """Host lifetime, open operations; no idle/finish decision envelopes."""
    def _context_inputs(self, context):
        return super()._context_inputs(replace(context, run_control={'stopping_owner': 'host'}))

    def _summary_current_state(self, context):
        return super()._summary_current_state(replace(context, run_control={'stopping_owner': 'host'}))

    def decision_schema(self, context):
        schema = super().decision_schema(context)
        schema['oneOf'] = [branch for branch in schema['oneOf']
            if branch['properties']['kind']['const'] not in ('wait', 'finish')]
        return schema

    def _decision_request(self, context):
        return replace(super()._decision_request(context), max_output_tokens=8192)

    def _parse_decision(self, response, context):
        decision = super()._parse_decision(response, context)
        if decision.kind not in ('invoke', 'explore'):
            raise ValueError('Select an available operation; the host owns stopping')
        return decision


class AllocatedProcessBackend(LocalProcessBackend):
    """Explicit host device/environment injection into scrubbed Lab children."""
    def __init__(self, root, environment):
        super().__init__(root, max_output_bytes=128000)
        self.environment = dict(environment)
        if not self.environment.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('An explicit allocated CUDA device mapping is required')
        if any(not isinstance(k, str) or not isinstance(v, str) or '=' in k or '\x00' in k + v
               for k, v in self.environment.items()):
            raise ValueError('Invalid host environment')

    def exec(self, sandbox, request):
        argv = ('/usr/bin/env', *(f'{k}={v}' for k, v in sorted(self.environment.items())), *request.argv)
        return super().exec(sandbox, replace(request, argv=argv))


EVALUATOR = r'''"""Editable reference: real HTTP inference, fixed development data, no training."""
import argparse, concurrent.futures, hashlib, json, re, time
from pathlib import Path
from urllib.request import Request, urlopen
p=argparse.ArgumentParser()
p.add_argument('--system', default='Choose the best answer. Return exactly one option letter, with no explanation.')
p.add_argument('--output', default='baseline-evaluation.json')
a=p.parse_args()
cfg=json.loads(Path('lab_config.json').read_text())
task=json.loads(Path('data/task.json').read_text())
def predict(row):
    prompt=row['question']+''.join('\n'+k+'. '+v for k,v in row['options'].items())
    body={'model':cfg['served_model'], 'messages':[{'role':'system','content':a.system},
        {'role':'user','content':prompt}], 'max_tokens':32, 'temperature':0,
        'chat_template_kwargs':{'enable_thinking':False}}
    with urlopen(Request(cfg['endpoint']+'/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'}),timeout=120) as r:
        result=json.load(r)
    if result['model'] != cfg['served_model']: raise ValueError('Provider identity mismatch')
    answer=result['choices'][0]['message']['content'].strip()
    match=re.fullmatch(r'([A-J])[.\s]*',answer)
    return {'id':row['id'],'gold':row['gold'],'prediction':match.group(1) if match else answer,
        'correct':bool(match and match.group(1)==row['gold']), 'usage':result.get('usage')}
metrics={}; predictions={}; inputs={}
for group in task['development']:
    path=Path('data')/group['path']; content=path.read_bytes()
    rows=[json.loads(line) for line in content.decode().splitlines() if line.strip()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(predict, rows))
    name=group['benchmark_id']; metrics[name]={'accuracy':sum(v['correct'] for v in values)/len(values),'n':len(values)}
    predictions[name]=values; inputs[name]=hashlib.sha256(content).hexdigest()
record={'kind':'development_evaluation','simulated':False,'at':time.time(),
    'candidate_id':'prompt:'+hashlib.sha256(a.system.encode()).hexdigest(),
    'base_model':cfg['model_path'],'weight_identity':cfg['weight_identity'],
    'protocol':'exact-option-letter-v1','system_prompt':a.system,'metrics':metrics,
    'dataset_sha256':inputs,'predictions':predictions,'adopted':False}
Path(a.output).write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='predictions'}),flush=True)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--python', required=True, help='Installed CUDA/Transformers Python for generated programs')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--max-calls', type=int, default=128)
    parser.add_argument('--deadline-epoch', type=float, help='Absolute host UTC deadline, never extended by initialization or restart')
    parser.add_argument('--max-jobs', type=int, default=1024)
    parser.add_argument('--program-timeout', type=int, default=1800)
    parser.add_argument('--revision', default='unreported')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    if min(args.seconds, args.max_calls, args.max_jobs, args.program_timeout) < 1:
        parser.error('Host limits must be positive')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.claim').open('x') as stream:
        stream.write(str(os.getpid()))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    binding = BackendBinding('meta', 'chat_completions', args.endpoint, timeout_seconds=120)
    model = ModelRef('meta', 'rsisys-meta-qwen35-9b', 'lion-local-qwen35-9b')
    transport = VLLMBackend(binding, generation_options={
        'temperature': 0.6, 'chat_template_kwargs': {'enable_thinking': False}}, native_schema_mode='portable')
    backend = RecordedBackend(transport, tokenizer, root, max_calls=args.max_calls,
                              template_tool_normalizer=sglang_template_tools)
    from .open_lab_monitor import OpenLabMonitor
    monitor = OpenLabMonitor(root, args.run_id, model, args.revision)
    previous_handlers = {}
    def request_stop(signum, frame):
        backend.stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    compactor = BasicContextCompactor(trigger_tokens=48000, target_tokens=16000, summary_tokens=2048,
        max_target_tokens=40000, target_increment_tokens=12000)
    prompts = DecisionPrompts({
        'control': 'The host owns stopping. Select and execute your next research action. Do not use wait or finish.',
        'scheduling': 'Use lab_poll for actual running programs, then lab_result to deliver terminal evidence. ',
    })
    task_path = Path(args.task).resolve()
    task = json.loads(task_path.read_text())
    # Do not discover or read any sibling heldout/private paths.
    relative_files = [task['train'], *(group['path'] for group in task['development'])]
    data_manifest = []
    for name in relative_files:
        path = (task_path.parent / name).resolve()
        if not path.is_relative_to(task_path.parent) or not path.is_file():
            raise ValueError('Task file escapes fixed input directory')
        content = path.read_bytes()
        data_manifest.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest(),
            'rows': len([line for line in content.splitlines() if line.strip()])})
    write(root / 'data-manifest.json', data_manifest)
    model_path = Path(args.model_path).resolve()
    weight_identity = {'model_path': str(model_path), 'config_sha256': hashlib.sha256(
        (model_path / 'config.json').read_bytes()).hexdigest(),
        'weight_shards': [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(model_path.glob('*.safetensors'))],
        'binding': 'local_config_and_weight_inventory; full_weight_digest_not_verified'}
    environment = {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}
    if os.environ.get('LD_LIBRARY_PATH'):
        environment['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src') + ':' +
        str(Path(__file__).resolve().parents[2]))
    write(root / 'backend-environment.json', environment)
    lab_endpoint = os.environ.get('RSISYS_LAB_ENDPOINT', args.endpoint)
    project = {'scenario': 'medical', 'python': args.python, 'model_path': args.model_path,
        'endpoint': lab_endpoint, 'served_model': model.model_id, 'task': 'data/task.json',
        'path_base': 'lab_workspace', 'data_root': 'data',
        'available_data': [{**entry, 'path': str(Path('data') / entry['path'])}
                           for entry in data_manifest],
        'editable_reference': 'reference_eval.py',
        'model_loading': 'Qwen3.5 uses transformers.AutoModelForImageTextToText, dtype=bfloat16, '
                         'attn_implementation=sdpa; use local_files_only=True. PEFT is installed.',
        'scientific_contract': 'Train only from task.train; development is for evaluation. '
            'Paths in task.json are relative to data_root; available_data paths are relative to the Lab workspace. '
            'Do not read or search for private/heldout data, install packages, or change external services. '
            'Lab code sees only the GPUs assigned to generated programs. '
            'The host separately allocates meta-agent and baseline inference services. '
            'You may write, repair and execute arbitrary research implementations within this Lab. '
            'The editable evaluator is optional and runs actual model inference. '
            'Preserve candidate identity, protocol, dataset identity and exact metrics. '
            'Pareto adoption requires primary improvement without guardrail regression; '
            'do not equate a program succeeding, a proposal, or unvalidated stdout with adoption. '
            'Development accuracy is not medical generalization.'}
    def make_lab(context):
        allocated = AllocatedProcessBackend(context.data_dir / 'allocated-workspaces', environment)
        context.add_owned_cleanup(allocated.close_all)
        lab = LocalProcessLab(allocated, context.artifacts, timeout_seconds=args.program_timeout,
                              max_jobs=args.max_jobs, max_parallel=1)
        # Seed assets before construction returns. No research programs are started.
        (lab.workspace / 'data').mkdir()
        for name in relative_files:
            target = lab.workspace / 'data' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(task_path.parent / name, target)
        shutil.copyfile(task_path, lab.workspace / 'data/task.json')
        (lab.workspace / 'reference_eval.py').write_text(EVALUATOR)
        write(lab.workspace / 'lab_config.json', {**project, 'weight_identity': weight_identity})
        return lab
    def make_decision(context, exploration):
        context.project_context = project
        context.backends = {'meta': backend}
        loop = build_default_decision(context, exploration)
        loop.harness.policy = TimedOpenPolicy(backend, model, allowed_actions=(),
            directives=DirectiveDefinitions(()), max_validation_attempts=3, structured_output=True,
            context_compactor=compactor, project_context=project, prompts=prompts)
        return loop
    from importlib import resources
    memory_instructions = resources.files('rsisys.loops.decision_making.feedback').joinpath(
        'prompts/langmem_research.txt').read_text()
    assembly = ResearchAssembly.langmem(memory_path=root / 'memory.sqlite3',
        namespace=('rsisys', args.run_id), memory_options={'max_calls_per_job': 6, 'max_steps': 1,
            'parallel_tool_calls': False,
            'additional_instructions': memory_instructions + '\nCapture concise reusable medical research and debugging observations. '
                'Make one focused tool call per reply. Keep each memory to a few sentences; '
                'do not copy code, complete prompts or unchanged environment descriptions into memory. '
                'Distinguish program outputs from validated evaluations and adoption. '
                'Absence from the supplied snapshot does not prove absence from the full history.'})
    modules = ModuleAssembly(lab_factory=make_lab, decision_factory=make_decision, research=assembly,
        research_backend=backend, feedback_options={'max_calls': args.max_calls, 'max_pending': 8,
            'max_input_tokens': 24000, 'max_output_tokens': 4096})
    objective = ('Improve the medical Lab using real evidence. Choose your own hypotheses and implement, '
        'execute, debug and compare them using the available tools. Read the installed task and environment '
        'information as needed. A submission is not completion: collect real terminal results with lab_result '
        'and use that feedback to choose your next action. Continue exploring; the host handles lifetime.')
    config = ApplicationConfig(objective, root / 'application', {'meta': binding}, {'decision_maker': model},
        execution='local_process', experience_limit=12, memory_limit=8, run_id=args.run_id)
    result = {'run_id': args.run_id, 'status': 'initializing', 'simulated': False,
        'weight_identity': weight_identity, 'memory_provider': 'langmem', 'started_at': None,
        'note': 'Model decisions and native program results are evidence; experiment/training counts require review.'}
    started = time.monotonic()
    try:
        remaining = min(args.seconds, args.deadline_epoch - time.time()) if args.deadline_epoch else args.seconds
        if remaining <= 0 or backend.stop_requested:
            result.update(status='deadline_reached_before_start')
            return result
        backend.deadline = time.monotonic() + remaining
        with Application(config, modules=modules, run_policy=AbsoluteHostPolicy(backend, args.seconds, args.deadline_epoch),
                         cleanup_timeout_seconds=40) as app:
            started = time.monotonic()
            result['started_at'] = time.time()
            remaining = min(args.seconds, args.deadline_epoch - time.time()) if args.deadline_epoch else args.seconds
            write(root / 'research-started.json', {'started_at': result['started_at'], 'seconds': remaining,
                'absolute_deadline_epoch': args.deadline_epoch})
            backend.deadline = started + max(0, remaining)
            try:
                while True:
                    if backend.stop_requested or time.monotonic() >= backend.deadline:
                        result.update(status='completed', stop='host_deadline_or_signal')
                        break
                    try:
                        step = app.runner.step(objective)
                    except HostAdmissionStop:
                        if backend.stop_requested:
                            result.update(status='completed', stop='host_signal')
                        elif backend.deadline is not None and time.monotonic() >= backend.deadline:
                            result.update(status='completed', stop='host_deadline')
                        else:
                            result.update(status='host_limit_reached', stop='inference_admission_limit')
                        break
                    append(root / 'steps.jsonl', {'at': time.time(), 'step': asdict(step)})
                    monitor.update(app.runner.decision_making.harness, backend, compactor, step)
                    write(root / 'context-audit.json', compactor.last_measurement)
                    if step.kind == 'finished':
                        result.update(status='completed', stop=step.event.get('run_stop'))
                        break
                    time.sleep(0.2)
            finally:
                lab = app.assembly.lab_execution
                until = time.monotonic() + 20
                while not lab.shutdown('cancel'):
                    if time.monotonic() > until:
                        raise TimeoutError('Owned Lab work did not stop')
                    time.sleep(0.1)
                # Retain generated programs/results/weights remotely; bundles may omit large/private files.
                shutil.copytree(lab.workspace, root / 'retained-workspace', symlinks=True,
                    ignore=lambda directory, names: {'data', '.cache'} & set(names))
                write(root / 'controller-cleanup.json', {'owned_lab_closed': True, 'at': time.time()})
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_at=time.time(), elapsed_seconds=time.monotonic() - started,
            model_calls=backend.calls, compactions=compactor.compactions)
        write(root / 'context-audit.json', compactor.last_measurement)
        monitor.finish(result)
        write(root / 'result.json', result)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
