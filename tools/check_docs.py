"""Check local file/heading links in maintained Markdown without network access."""
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text):
    lines, fence = [], None
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            value = marker[1]
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
        elif fence is None:
            lines.append(line)
    return '\n'.join(lines)


def anchors(path):
    text = prose(path.read_text())
    found, seen = set(), {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        key = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        index = seen.get(key, 0)
        seen[key] = index + 1
        found.add(key if index == 0 else f'{key}-{index}')
    found.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    return found


def main():
    paths = list(ROOT.glob('*.md'))
    for name in ('docs', 'examples', 'experiments', 'src'):
        paths.extend((ROOT / name).rglob('*.md'))
    errors, checked, cache = [], 0, {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        if (relative.parts[:2] == ('docs', 'review') or
                relative == Path('docs/development-history.md') or
                any(part in {'__pycache__', 'build', 'dist', '.venv'} or part.endswith('.egg-info')
                    for part in relative.parts)):
            continue
        text = prose(path.read_text())
        for raw in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
            link = urlsplit(raw.strip('<>'))
            if link.scheme or link.netloc:
                continue
            target = (path.parent / unquote(link.path)).resolve() if link.path else path
            checked += 1
            if not target.exists():
                errors.append(f'{relative}: missing {raw}')
            elif link.fragment and target.suffix == '.md':
                if target not in cache:
                    cache[target] = anchors(target)
                if unquote(link.fragment) not in cache[target]:
                    errors.append(f'{relative}: unknown heading {raw}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Validated {checked} local documentation links (historical reviews excluded).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
