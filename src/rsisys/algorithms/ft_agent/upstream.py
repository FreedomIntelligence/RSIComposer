"""Pinned FT-Agent policy code with explicit local execution/transport adapters.

Only named definitions are compiled from the verified checkout. This avoids
import-time provisioning of all FT-Dojo datasets, Docker and unrelated agents.
Hypothesis generation, feedback decisions and trace traversal remain upstream
code. Workspace, scenario, API transport and execution are adapter boundaries.
"""
import __future__
import ast
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
import inspect
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, Generic, TypeVar

from rsisys.runtime.upstream import checkout

FT_AGENT_REVISION = '484776c211e4fbbeef03e0ec00d6bbee7362a4f4'
LLAMA_FACTORY_REVISION = 'ca75f1edf3cb50343ed1c98605141c3e22075b5f'


class Workspace:
    """In-memory file exchange; generated code executes only inside the Lab."""
    def __init__(self):
        self.file_dict = {}
        self.running_info = SimpleNamespace(result=None, running_time=None)

    def inject_from_workspace(self, other):
        self.file_dict = deepcopy(other.file_dict)


class Template:
    def __init__(self, uri):
        import yaml
        from jinja2 import Environment, StrictUndefined
        caller = Path(inspect.currentframe().f_back.f_code.co_filename).parent
        name, key = uri.split(':')
        if name != '.prompts':
            raise ValueError('Only pinned adjacent FT-Agent templates are supported')
        value = yaml.safe_load((caller / 'prompts.yaml').read_text())
        for part in key.split('.'):
            value = value[part]
        # Same undefined behavior and default filters as the upstream RDAT.
        self.template = Environment(undefined=StrictUndefined).from_string(value)

    def r(self, **values):
        rendered = self.template.render(**values).strip('\n')
        while '\n\n\n' in rendered:
            rendered = rendered.replace('\n\n\n', '\n\n')
        return rendered


class Scenario:
    def __init__(self, description, baseline):
        self.description = description
        self.baseline_benchmark_score = baseline
        self.selected_datasets = ['seed']
        self.model_info = {'has_think_token': False}

    def get_scenario_all_desc(self, **kwargs):
        return self.description


def load_policy(root, *, settings, backend, parameter_info, dataset_descriptor):
    import yaml
    root = checkout(root, FT_AGENT_REVISION)
    logger = logging.getLogger('rsisys.ft_agent')
    namespace = dict(ABC=ABC, abstractmethod=abstractmethod, Any=Any, Dict=Dict,
        Generic=Generic, Sequence=Sequence, Path=Path, json=json, re=re, yaml=yaml,
        RunningInfo=SimpleNamespace, FTWorkspace=Workspace, T=Template, KnowledgeBase=object,
        logger=logger, FT_RD_SETTING=settings, APIBackend=lambda: backend,
        dict_get_with_warning=lambda obj, key, default: obj.get(key, default),
        FT_YAML_FILE_NAME='train.yaml', FT_DATA_SCRIPT_NAME='process_data.py',
        FT_TEST_PARAMS_FILE_NAME='test_params.yaml',
        DEFAULT_HELP_TRUNCATE_LEN=None, EXCLUDED_PARAM_REGEX=re.compile(r'(?!)'),
        FinetuneDatasetDescriptor=lambda: dataset_descriptor,
        FT_PATHS=SimpleNamespace(datasets='/datasets/', models='/models/', workspace='./', deepspeed=''))
    for name in ('ASpecificTask', 'ASpecificWSForExperiment', 'ASpecificWSForSubTasks',
                 'ASpecificScen', 'ASpecificKB'):
        namespace[name] = TypeVar(name)

    def load(file, names):
        source = root / 'rdagent' / file
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError('Pinned upstream definitions are missing: ' + file)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)

    load('core/evaluation.py', ['Feedback'])
    load('utils/__init__.py', ['convert2bool'])
    load('core/experiment.py', ['AbsTask', 'Task', 'UserInstructions', 'Experiment'])
    load('core/proposal.py', ['Hypothesis', 'ExperimentFeedback', 'HypothesisFeedback',
                           'Trace', 'ExpGen', 'Experiment2Feedback'])
    load('components/coder/CoSTEER/task.py', ['CoSTEERTask'])
    load('components/coder/finetune/exp.py', ['FTTask'])
    load('scenarios/finetune/experiment/experiment.py', ['FTExperiment'])
    load('scenarios/finetune/proposal/trace.py', ['FTTrace'])
    load('scenarios/finetune/scen/llama_factory_manager.py', ['LLaMAFactoryManager'])
    manager = namespace['LLaMAFactoryManager'].__new__(namespace['LLaMAFactoryManager'])
    manager._info_cache = parameter_info
    namespace['LLaMAFactory_manager'] = manager

    def ensure_assets(**kwargs):
        if kwargs.get('model') != settings.base_model:
            raise ValueError('FT-Agent requested an unconfigured model')
    namespace['ensure_ft_assets_exist'] = ensure_assets
    load('scenarios/finetune/proposal/proposal.py', ['FTHypothesis', 'LLMFinetuneExpGen'])
    load('scenarios/finetune/dev/feedback.py', ['FTExperiment2Feedback'])

    class StrategyBase:
        def __init__(self, scen, settings):
            self.scen, self.settings = scen, settings
    namespace['MultiProcessEvolvingStrategy'] = StrategyBase
    load('components/coder/finetune/__init__.py', ['LLMFinetuneEvolvingStrategy'])
    return SimpleNamespace(**namespace)


class ChatBackend:
    """OpenAI-compatible transport for upstream APIBackend call sites."""
    def __init__(self, config, directory):
        from rsisys.labs.harness.runtime import ChatPool
        self.pool = ChatPool(config['endpoints'], config['model'],
            timeout=config.get('timeout', 600), max_tokens=config.get('max_tokens', 8192),
            temperature=config.get('temperature', 0.6))
        self.root, self.calls = Path(directory), 0

    def build_chat_session(self, session_system_prompt):
        backend = self
        class Session:
            def __init__(self):
                self.messages = [{'role': 'system', 'content': session_system_prompt}]

            def build_chat_completion(self, user_prompt, **kwargs):
                self.messages.append({'role': 'user', 'content': user_prompt})
                content = backend.complete(self.messages, **kwargs)
                self.messages.append({'role': 'assistant', 'content': content})
                return content
        return Session()

    def build_messages_and_create_chat_completion(self, user_prompt, system_prompt, **kwargs):
        return self.complete([{'role': 'system', 'content': system_prompt},
                              {'role': 'user', 'content': user_prompt}], **kwargs)

    def complete(self, messages, **kwargs):
        self.calls += 1
        result = self.pool.complete(messages, seed=7308 + self.calls,
            response_schema={'type': 'object', 'additionalProperties': True}
            if kwargs.get('json_mode') or kwargs.get('json_target_type') else None)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f'call-{self.calls:04d}.json').write_text(json.dumps(
            {'messages': messages, 'options': {k: str(v) for k, v in kwargs.items()},
             'response': result}, ensure_ascii=False, indent=2))
        if result['finish_reason'] != 'stop':
            raise ValueError('FT-Agent generation was incomplete: ' + str(result['finish_reason']))
        content = result['content']
        language = kwargs.get('code_block_language')
        if language:
            matches = re.findall(r'```' + re.escape(language) + r'\s*\n(.*?)```', content, re.S)
            if len(matches) != 1:
                raise ValueError('FT-Agent must return exactly one ' + language + ' code block')
            content = matches[0]
        if kwargs.get('json_mode') or kwargs.get('json_target_type'):
            if not isinstance(json.loads(content), dict):
                raise ValueError('FT-Agent JSON response must be an object')
        return content
