"""Run the unmodified author executor only inside the host-provided OS sandbox."""
import json
from pathlib import Path
import sys

directory, source = map(Path, sys.argv[1:3])
sys.path.insert(0, str(source))
from executors.py_executor import PyExecutor

spec = json.loads((directory / 'input.json').read_text())
executor = PyExecutor()
if spec['mode'] == 'execute':
    result = executor.execute(spec['func'], spec['tests'], spec['timeout'])
else:
    result = executor.evaluate(spec['name'], spec['func'], spec['test'], spec['timeout'])
# SystemExit or os._exit before the executor returns cannot produce this receipt.
(directory / 'result.json').write_text(json.dumps(result))
