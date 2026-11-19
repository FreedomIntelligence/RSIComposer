"""Drop unused English entries from messages.js and add the given new ones.

Usage: python3 i18n_sync.py <frontend-dir> <additions.json>
"""
import json
import pathlib
import re
import subprocess
import sys

frontend = pathlib.Path(sys.argv[1])
additions = json.loads(pathlib.Path(sys.argv[2]).read_text())
path = frontend / 'app/messages.js'
text = path.read_text()
known = json.loads(text[text.index('{'):text.rindex('}') + 1])
report = subprocess.run([sys.executable, str(frontend / 'scripts/i18n_keys.py'), str(frontend)],
                        capture_output=True, text=True, check=True).stdout
unused = set(json.loads(report.split('UNUSED\n')[1]))
merged = {key: value for key, value in {**known, **additions}.items() if key not in unused or key in additions}
lines = ',\n'.join(f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}' for key, value in sorted(merged.items()))
path.write_text(text[:text.index('{')] + '{\n' + lines + '\n};\n')
print(len(known), '->', len(merged))
