"""List UI strings (Chinese source keys) used by the monitor app and report missing English entries.

Usage: python3 i18n_keys.py <frontend-dir> [legacy-i18n.js]
"""
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
files = [root / 'App.vue', *sorted((root / 'pages').glob('*.vue')), *sorted((root / 'components').glob('*.vue')),
         root / 'app/store.js', root / 'app/i18n.js', root / 'app/insight.js']
CJK = re.compile('[一-鿿]')
LITERAL = re.compile(r"'([^'\n]*)'")  # UI keys are always single-quoted, also inside templates
keys = set()
for path in files:
    for match in LITERAL.finditer(path.read_text()):
        text = match.group(1)
        if CJK.search(text):
            keys.add(text)

known = {}
messages = root / 'app/messages.js'
if messages.exists():
    body = messages.read_text()
    known = json.loads(body[body.index('{'):body.rindex('}') + 1])
if len(sys.argv) > 2:
    old = pathlib.Path(sys.argv[2]).read_text()
    start = old.index('const english = Object.freeze(') + len('const english = Object.freeze(')
    legacy = json.loads(old[start:old.index('} );') + 1])
    known = {**{key: legacy[key] for key in keys if key in legacy}, **known}

print(json.dumps({key: known[key] for key in sorted(keys) if key in known}, ensure_ascii=False, indent=1))
print('MISSING')
print(json.dumps(sorted(keys - set(known)), ensure_ascii=False))
print('UNUSED')
print(json.dumps(sorted(set(known) - keys), ensure_ascii=False))
