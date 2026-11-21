"""Score retrieval agents with the BrowseComp and WideSearch graders."""

import json
from pathlib import Path
import subprocess

from rsisys.labs._task_evaluation import TaskEvaluationLab, checked_file
from rsisys.loops.contracts.benchmark import canonical_json


class SearchLab(TaskEvaluationLab):
    domain = 'search'
    _settings = {'checkout', 'revision', 'python', 'judge_endpoint', 'judge_model', 'judge_key_env'}
    adapters = {
        'browsecomp': ('accuracy', 'openai-browsecomp/v1', _settings),
        'widesearch': ('f1_by_item', 'bytedance-widesearch/v1', _settings),
    }

    def measure(self, benchmark, task, instruction):
        entry = self.entries[benchmark]
        output = self.runner(instruction=instruction, task=dict(task['input']), context=self.context)
        if (not isinstance(output, dict) or not isinstance(output.get('response'), str)
                or not isinstance(output.get('sources', []), list)):
            raise ValueError('Search runner must return response text and optionally a list of sources')
        reference = dict(task['reference'])
        if benchmark == 'widesearch':
            root = Path(entry['manifest']).resolve().parent
            path = (root / reference.pop('answer_file')).resolve()
            if not path.is_relative_to(root):
                raise ValueError('WideSearch gold tables must stay inside the manifest directory')
            reference['answer_csv'] = checked_file(path, reference.pop('sha256')).decode()
        worker = Path(__file__).with_name('_grader.py')
        settings = {key: entry[key] for key in self._settings if key != 'python'}
        request = {'benchmark': benchmark, 'settings': settings, 'input': task['input'],
                   'id': task['id'], 'reference': reference, 'response': output['response']}
        result = subprocess.run([entry['python'], str(worker)], input=canonical_json(request),
                                capture_output=True, text=True, timeout=900, check=True)
        return json.loads(result.stdout), output
