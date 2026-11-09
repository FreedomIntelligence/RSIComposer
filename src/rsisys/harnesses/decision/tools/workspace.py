"""Explicit, bounded local workspace observations; not an OS security sandbox."""

import codecs
import json
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.decision_making.action_space.capabilities import (
    Capabilities,
    Capability,
)


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _json_bytes(data):
    return len(json.dumps(data, ensure_ascii=True).encode('utf-8'))


class WorkspaceTools:
    """Expose only caller-selected named directories through ordinary capabilities.

    ``max_bytes`` bounds each file read, ``scan_budget_bytes`` bounds all file
    content read by one search, and ``max_entries`` bounds directory traversal.
    No roots, credentials, environment variables or service state are discovered.
    """

    def __init__(self, roots: Mapping[str, str | os.PathLike], *, max_bytes=64 * 1024,
                 max_output_bytes=128 * 1024, scan_budget_bytes=1024 * 1024,
                 max_entries=1000, max_depth=8, max_results=100):
        if not isinstance(roots, Mapping) or not roots:
            raise ValueError('Supply at least one explicitly named workspace root')
        self._roots = {}
        for name, path in roots.items():
            if (not isinstance(name, str) or not name or len(name) > 64
                    or not all(c.isascii() and (c.isalnum() or c in '_-') for c in name)):
                raise ValueError('Root names must contain 1–64 ASCII letters, digits, _ or -')
            resolved = Path(path).resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError('Workspace roots must be existing directories')
            self._roots[name] = resolved
        self.max_bytes = _integer(max_bytes, 'max_bytes', 1, 1024 ** 3)
        self.max_output_bytes = _integer(max_output_bytes, 'max_output_bytes', 4096, 1024 ** 3)
        self.scan_budget_bytes = _integer(scan_budget_bytes, 'scan_budget_bytes', 1, 1024 ** 3)
        self.max_entries = _integer(max_entries, 'max_entries', 1, 10 ** 6)
        self.max_depth = _integer(max_depth, 'max_depth', 0, 64)
        self.max_results = _integer(max_results, 'max_results', 1, 10000)
        # Fail closed on platforms without descriptor-relative no-follow opens.
        if not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
            raise NotImplementedError('WorkspaceTools requires descriptor-relative O_NOFOLLOW opens')
        self.capabilities = Capabilities(self._capability(kind) for kind in ('list', 'read', 'search'))

    @property
    def definitions(self):
        return self.capabilities.definitions

    @property
    def bindings(self):
        return self.capabilities.bindings

    def describe(self):
        return tuple(self.capabilities)

    def _capability(self, kind):
        properties = {
            'root': {'type': 'string', 'enum': list(self._roots)},
            'path': {'type': 'string', 'minLength': 1, 'maxLength': 256, 'default': '.',
                     'description': 'Relative to the named root; absolute paths and .. are rejected.'},
            'offset': {'type': 'integer', 'minimum': 0, 'maximum': 10 ** 9, 'default': 0},
        }
        if kind in ('list', 'search'):
            properties['limit'] = {'type': 'integer', 'minimum': 1, 'maximum': self.max_results}
        if kind in ('read', 'search'):
            properties['max_bytes'] = {'type': 'integer', 'minimum': 1, 'maximum': self.max_bytes}
        if kind == 'search':
            properties.update(
                query={'type': 'string', 'minLength': 1, 'maxLength': 256, 'pattern': r'^[^\r\n]+$'},
                scan_budget_bytes={'type': 'integer', 'minimum': 1, 'maximum': self.scan_budget_bytes},
                max_entries={'type': 'integer', 'minimum': 1, 'maximum': self.max_entries},
                max_depth={'type': 'integer', 'minimum': 0, 'maximum': self.max_depth},
            )
        description = {
            'list': 'List a workspace directory in filesystem order. Offset counts directory entries; '
                    'follow next_offset. Listing is bounded and is not a snapshot.',
            'read': 'Read a bounded UTF-8 text window. Offset and next_offset count bytes; '
                    'UTF-8 characters split by a byte boundary use replacement characters. Binary is unavailable.',
            'search': 'Search a file or bounded directory for a case-sensitive literal UTF-8 query. '
                      'Offset counts matching lines in the bounded scan, not bytes. Each file is scanned '
                      'only up to max_bytes. Inspect truncated/reasons and narrow the path for incomplete scans.',
        }[kind]
        spec = OperationSpec('workspace_' + kind, description, {
            'type': 'object', 'properties': properties,
            'required': ['root', 'query'] if kind == 'search' else ['root'],
            'additionalProperties': False,
        }, read_only=True)
        return Capability.from_spec(spec, partial(self._validate, kind), partial(self._execute, kind))

    def _validate(self, kind, arguments):
        allowed = {'root', 'path', 'offset'}
        if kind in ('list', 'search'):
            allowed.add('limit')
        if kind in ('read', 'search'):
            allowed.add('max_bytes')
        if kind == 'search':
            allowed.update(('query', 'scan_budget_bytes', 'max_entries', 'max_depth'))
        if not isinstance(arguments, dict) or set(arguments) - allowed:
            raise ValueError('Unknown workspace arguments')
        root = arguments.get('root')
        if not isinstance(root, str) or root not in self._roots:
            raise ValueError('Select an explicitly configured root')
        path = arguments.get('path', '.')
        if (not isinstance(path, str) or not path or len(path) > 256 or '\x00' in path
                or Path(path).is_absolute() or '..' in Path(path).parts):
            raise ValueError('path must be a relative workspace path without .. (maximum 256 characters)')
        args = {'root': root, 'path': Path(path).as_posix(),
                'offset': _integer(arguments.get('offset', 0), 'offset', 0, 10 ** 9)}
        for key, default, minimum, maximum in (
            ('limit', min(20, self.max_results), 1, self.max_results),
            ('max_bytes', self.max_bytes, 1, self.max_bytes),
            ('scan_budget_bytes', self.scan_budget_bytes, 1, self.scan_budget_bytes),
            ('max_entries', self.max_entries, 1, self.max_entries),
            ('max_depth', self.max_depth, 0, self.max_depth),
        ):
            if key in allowed:
                args[key] = _integer(arguments.get(key, default), key, minimum, maximum)
        if kind == 'search':
            query = arguments.get('query')
            if not isinstance(query, str) or not query or len(query) > 256 or '\n' in query or '\r' in query:
                raise ValueError('query must be a nonempty single-line literal of at most 256 characters')
            args['query'] = query
        return OperationCall('workspace_' + kind, args)

    @contextmanager
    def _open(self, root, relative):
        """Resolve in-root links, then walk canonical components without following links.

        A link swapped between resolution and open fails closed, including ancestor
        links. Directory symlinks are not recursively traversed by search.
        """
        base = self._roots[root]
        target = (base / relative).resolve(strict=True)
        if not target.is_relative_to(base):
            raise PermissionError('Path leaves the selected workspace root')
        fd = os.open(target.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            parts = target.parts[1:]
            for index, part in enumerate(parts):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if index < len(parts) - 1:
                    flags |= os.O_DIRECTORY
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            info = os.fstat(fd)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise PermissionError('Only regular files and directories are supported')
            yield fd, info
        finally:
            os.close(fd)

    def _execute(self, kind, arguments):
        args = self._validate(kind, arguments).arguments
        data = dict(source='workspace:' + args['root'], root=args['root'], path=args['path'],
                    observed_at=datetime.now(timezone.utc).isoformat(), read_only=True,
                    available=True, truncated=False, reasons=[], offset=args['offset'], next_offset=None)
        try:
            with self._open(args['root'], args['path']) as (fd, info):
                getattr(self, '_' + kind)(args, data, fd, info)
        except (OSError, RuntimeError) as exc:
            # Do not expose host paths (including symlink targets) in exception text.
            reason = ('not_found' if isinstance(exc, FileNotFoundError) else
                      'not_directory' if isinstance(exc, NotADirectoryError) else
                      'path_denied' if isinstance(exc, PermissionError) else 'unavailable')
            data.update(available=False, unavailable_reason=reason)
        assert _json_bytes(data) <= self.max_output_bytes
        return OperationResult(f'Workspace {kind}: {args["root"]}/{args["path"]}', (), False, data=data)

    @staticmethod
    def _truncate(data, reason):
        data['truncated'] = True
        if reason not in data['reasons']:
            data['reasons'].append(reason)

    def _append(self, data, item):
        # Reserve space for counters, continuation and truncation facts added later.
        if _json_bytes(data) + _json_bytes(item) + 512 > self.max_output_bytes:
            self._truncate(data, 'output_budget')
            return False
        data['items'].append(item)
        return True

    def _list(self, args, data, fd, info):
        if not stat.S_ISDIR(info.st_mode):
            data.update(available=False, unavailable_reason='not_directory')
            return
        data.update(items=[], scanned_entries=0)
        with os.scandir(fd) as entries:
            for index, entry in enumerate(entries):
                if data['scanned_entries'] >= self.max_entries:
                    self._truncate(data, 'entry_budget')
                    # An offset larger than the budget cannot be reached by rescanning.
                    break
                data['scanned_entries'] += 1
                if index < args['offset']:
                    continue
                if len(data['items']) >= args['limit']:
                    data['next_offset'] = index
                    self._truncate(data, 'page_limit')
                    break
                path = (Path(args['path']) / entry.name).as_posix()
                if len(path) > 256:
                    self._truncate(data, 'path_length')
                    continue
                try:
                    with self._open(args['root'], path) as (_, child):
                        item = {'path': path, 'kind': 'directory' if stat.S_ISDIR(child.st_mode) else 'file',
                                'size_bytes': child.st_size, 'symlink': entry.is_symlink()}
                except (OSError, RuntimeError):
                    # Never disclose a link target, nor follow it outside this root.
                    item = {'path': path, 'kind': 'unavailable', 'available': False}
                if not self._append(data, item):
                    data['next_offset'] = index
                    break

    @staticmethod
    def _binary(raw, *, offset=0, has_more=False):
        if any(byte < 32 and byte not in (9, 10, 13) for byte in raw):
            return True
        # An arbitrary byte offset may start in a valid UTF-8 continuation sequence.
        prefix = 0
        if offset:
            while prefix < min(3, len(raw)) and 0x80 <= raw[prefix] <= 0xBF:
                prefix += 1
        try:
            codecs.getincrementaldecoder('utf-8')().decode(raw[prefix:], final=not has_more)
        except UnicodeDecodeError:
            return True
        return False

    def _read(self, args, data, fd, info):
        if not stat.S_ISREG(info.st_mode):
            data.update(available=False, unavailable_reason='not_file')
            return
        # JSON escaping can expand one input byte to six output bytes.
        count = min(args['max_bytes'], max(1, (self.max_output_bytes - _json_bytes(data) - 512) // 6))
        os.lseek(fd, args['offset'], os.SEEK_SET)
        raw = os.read(fd, count)
        end = args['offset'] + len(raw)
        data.update(size_bytes=info.st_size, bytes_read=len(raw), encoding='utf-8',
                    decoding_errors='replace', binary=self._binary(
                        raw, offset=args['offset'], has_more=end < info.st_size))
        if end < info.st_size:
            self._truncate(data, 'byte_limit' if count == args['max_bytes'] else 'output_budget')
            data['next_offset'] = end
        if data['binary']:
            data.update(available=False, unavailable_reason='binary')
        else:
            data['text'] = raw.decode('utf-8', errors='replace')

    def _search(self, args, data, fd, info):
        data.update(items=[], scanned_bytes=0, scanned_entries=0, scanned_files=0,
                    skipped_files=0, matched_lines=0)
        query = args['query'].encode('utf-8')

        def scan(path, file_fd, file_info):
            remaining = args['scan_budget_bytes'] - data['scanned_bytes']
            if remaining <= 0:
                self._truncate(data, 'scan_budget')
                return False
            raw = os.read(file_fd, min(args['max_bytes'], remaining))
            data['scanned_files'] += 1
            data['scanned_bytes'] += len(raw)
            if len(raw) < file_info.st_size:
                self._truncate(data, 'scan_budget' if remaining < args['max_bytes'] else 'file_byte_limit')
            if self._binary(raw, has_more=len(raw) < file_info.st_size):
                data['skipped_files'] += 1
                return True
            start = 0
            for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
                position = line.find(query)
                if position >= 0:
                    index = data['matched_lines']
                    data['matched_lines'] += 1
                    if index >= args['offset']:
                        if len(data['items']) >= args['limit']:
                            data['next_offset'] = index
                            self._truncate(data, 'page_limit')
                            return False
                        # Bounded excerpt centered on the match, including the full query.
                        left = max(0, position - 128)
                        right = min(len(line), position + len(query) + 256)
                        item = {'path': path, 'line': line_number, 'byte_offset': start + position,
                                'text': line[left:right].decode('utf-8', errors='replace'),
                                'truncated': left > 0 or right < len(line)}
                        if not self._append(data, item):
                            data['next_offset'] = index
                            return False
                start += len(line)
            return True

        def walk(path, directory_fd, depth):
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if data['scanned_entries'] >= args['max_entries']:
                        self._truncate(data, 'entry_budget')
                        return False
                    if data['scanned_bytes'] >= args['scan_budget_bytes']:
                        self._truncate(data, 'scan_budget')
                        return False
                    data['scanned_entries'] += 1
                    child_path = (Path(path) / entry.name).as_posix()
                    if len(child_path) > 256:
                        self._truncate(data, 'path_length')
                        continue
                    try:
                        with self._open(args['root'], child_path) as (child_fd, child_info):
                            if stat.S_ISDIR(child_info.st_mode):
                                if entry.is_symlink():
                                    self._truncate(data, 'directory_symlink')
                                elif depth >= args['max_depth']:
                                    self._truncate(data, 'depth_budget')
                                elif not walk(child_path, child_fd, depth + 1):
                                    return False
                            elif not scan(child_path, child_fd, child_info):
                                return False
                    except (OSError, RuntimeError):
                        data['skipped_files'] += 1
                        self._truncate(data, 'unavailable_path')
            return True

        if stat.S_ISREG(info.st_mode):
            scan(args['path'], fd, info)
        else:
            walk(args['path'], fd, 0)
