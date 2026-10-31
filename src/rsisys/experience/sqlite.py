"""Durable system feedback, independent of loop-local task memory."""

from dataclasses import asdict
from pathlib import Path
import json
import sqlite3

from rsisys.runtime.files import json_bytes
from .records import ExperienceRecord, Scope


def decode_record(payload: str) -> ExperienceRecord:
    data = json.loads(payload)
    data['source_refs'] = tuple(data['source_refs'])
    return ExperienceRecord(**data)


def validate_query(scope, limit):
    if scope not in ('actor', 'decision_maker', 'shared', None):
        raise ValueError('Unknown experience scope')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')


class SQLiteExperienceStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experience (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL, scope TEXT NOT NULL,
                kind TEXT NOT NULL, summary TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS experience_scope ON experience(scope, seq);
        ''')
        row = self.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if row is not None and row[0] != '2':
            self.connection.close()
            raise ValueError('Unsupported experience schema version')
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version', '2')")

    def append(self, record: ExperienceRecord) -> None:
        payload = json_bytes(asdict(record)).decode()
        with self.connection:
            existing = self.connection.execute('SELECT payload FROM experience WHERE record_id=?',
                                                (record.record_id,)).fetchone()
            if existing is not None:
                if existing[0] == payload:
                    return
                raise ValueError('Experience identity is already bound to different content')
            self.connection.execute('INSERT INTO experience(record_id,scope,kind,summary,payload) VALUES (?,?,?,?,?)',
                                    (record.record_id, record.scope, record.kind, record.summary, payload))

    def records(self) -> tuple[ExperienceRecord, ...]:
        return self.query()

    def query(self, scope: Scope | None = None, *, limit: int | None = None,
              kinds: tuple[str, ...] = (), text: str | None = None) -> tuple[ExperienceRecord, ...]:
        validate_query(scope, limit)
        clauses, parameters = [], []
        if scope is not None:
            clauses.append('scope IN (?, ?)')
            parameters.extend((scope, 'shared'))
        if kinds:
            clauses.append('kind IN (' + ','.join('?' for _ in kinds) + ')')
            parameters.extend(kinds)
        if text:
            clauses.append('instr(lower(summary), lower(?)) > 0')
            parameters.append(text)
        sql = 'SELECT payload FROM experience'
        if clauses:
            sql += ' WHERE ' + ' AND '.join(clauses)
        sql += ' ORDER BY seq DESC'
        if limit is not None:
            sql += ' LIMIT ?'
            parameters.append(limit)
        rows = self.connection.execute(sql, parameters).fetchall()
        return tuple(decode_record(row[0]) for row in reversed(rows))

    def backup(self, path: Path) -> None:
        target = sqlite3.connect(path)
        try:
            self.connection.backup(target)
        finally:
            target.close()

    def close(self) -> None:
        self.connection.close()
