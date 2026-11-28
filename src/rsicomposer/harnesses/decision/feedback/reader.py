"""Bounded, read-only pages of recorded feedback evidence and proposals."""
from copy import deepcopy

from rsicomposer.harnesses.decision.feedback.knowledge import (
    interpretation_page,
    proposal_records,
)
from rsicomposer.runtime.storage.files import json_bytes
from rsicomposer.runtime.storage.json_tools import (
    EvidencePathError,
    encoded,
    evidence_value,
    result_id,
)

_EVIDENCE_PAGE_BYTES = 6000


class FeedbackReader:
    def read(self, job_id=None, *, offset=0, limit=8, research_attempt_id=None, evidence_path=None,
             view='jobs', target=None, expected_digest=None):
        """Read proposals, or a bounded JSON-text page of original evidence.

        evidence_path is a JSON pointer into raw_input (empty for the root).
        /dispatch_attempts addresses the separately refreshed catalog. Evidence
        offsets count characters, with at most 128 characters per limit unit.
        Pages also fit 6000 JSON-encoded UTF-8 bytes including metadata (except
        irreducible metadata plus one character). Assemble all pages before
        parsing their JSON. Reads never expand model coverage.
        """
        self._assert_owner()
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError('Invalid learning page')
        if view not in ('jobs', 'attempts', 'proposals', 'trajectories'):
            raise ValueError('Unknown research feedback view')
        if expected_digest is not None and (not isinstance(expected_digest, str) or len(expected_digest) != 64):
            raise ValueError('expected_digest must be a SHA-256 digest')
        if view != 'jobs':
            if job_id is not None or evidence_path is not None:
                raise ValueError('Knowledge views cannot be combined with a job evidence read')
            if research_attempt_id is not None and (not isinstance(research_attempt_id, str)
                    or not research_attempt_id.strip() or len(research_attempt_id) > 200):
                raise ValueError('Invalid research attempt ID')
            if view == 'trajectories':
                if target is not None:
                    raise ValueError('target applies only to proposal views')
                value = self.trajectories(research_attempt_id=research_attempt_id)
            elif view == 'attempts':
                if target is not None:
                    raise ValueError('target applies only to proposal views')
                value = self.attempts()['items']
                if research_attempt_id is not None:
                    value = [a for a in value if a['attempt_id'] == research_attempt_id]
            else:
                value = proposal_records(self.ordered_jobs(), targets=self.proposal_targets, target=target,
                                         research_attempt_id=research_attempt_id)
            metadata = {'view': view, 'total': len(value),
                        'target': target, 'research_attempt_id': research_attempt_id}
            if view == 'trajectories':
                metadata['enabled'] = self.trajectory_archive is not None
            return interpretation_page(value, offset=offset, limit=limit,
                expected_digest=expected_digest, metadata=metadata,
                authority='archived_public_trajectory' if view == 'trajectories' else 'unverified_interpretations')
        if target is not None:
            raise ValueError('target applies only to proposal views')
        if research_attempt_id is not None and (not isinstance(research_attempt_id, str)
                or not research_attempt_id.strip() or len(research_attempt_id) > 200 or job_id is not None):
            raise ValueError('An attempt filter requires a bounded ID and a job listing')
        if evidence_path is not None:
            if (job_id is None or not isinstance(evidence_path, str) or len(evidence_path) > 1000
                    or (evidence_path and not evidence_path.startswith('/'))):
                raise ValueError('Evidence reading requires a job ID and a JSON pointer of at most 1000 characters')
            if job_id not in self.state['jobs']:
                raise EvidencePathError('Evidence job ID is not recorded; list feedback jobs before reading evidence')
            row = self.state['jobs'][job_id]
            if evidence_path.split('/')[1:2] in (['context_memory'], ['trajectory'], ['trajectory_snapshot']):
                namespace = evidence_path.split('/')[1]
                if namespace == 'context_memory':
                    value = self.context_record(job_id)
                elif namespace == 'trajectory_snapshot':
                    value = self.trajectory_snapshot(job_id)
                else:
                    if self.trajectory_archive is None:
                        raise EvidencePathError('Trajectory archival is disabled')
                    records = self._trajectory_records([row])
                    if len(records) != 1:
                        raise EvidencePathError('No unique trajectory segment for this job')
                    value = records[0]
                value = evidence_value({namespace: value}, evidence_path)
                return interpretation_page(value, offset=offset, limit=limit,
                    expected_digest=expected_digest, authority=(
                        'organized_context_memory' if namespace == 'context_memory' else 'archived_public_trajectory'),
                    metadata={'job_id': job_id, 'evidence_path': evidence_path, 'simulated': row['simulated']})
            if evidence_path.split('/')[1:2] in (['proposals'], ['attempt_revisions'], ['outputs'], ['output']):
                # Interpretation output has its own namespace. It is never
                # presented as captured execution evidence or admitted coverage.
                value = evidence_value(row, evidence_path)
                return interpretation_page(value, offset=offset, limit=limit,
                    expected_digest=expected_digest, metadata={'job_id': job_id,
                        'evidence_path': evidence_path, 'simulated': row['simulated']})
            value = row['raw_input']
            root = '/raw_input'
            pointer = evidence_path
            if pointer == '/dispatch_attempts' or pointer.startswith('/dispatch_attempts/'):
                value = row['dispatch_attempts']
                root = '/dispatch_attempts'
                pointer = pointer[len('/dispatch_attempts'):]
            for namespace, key in (('/feedback_request', 'request'), ('/feedback_response', 'response')):
                if pointer == namespace or pointer.startswith(namespace + '/'):
                    if row.get(key) is None:
                        raise EvidencePathError('Feedback exchange was not recorded for this job')
                    value, root, pointer = row[key], '/' + key, pointer[len(namespace):]
                    break
            value = evidence_value(value, pointer)
            if expected_digest is not None and expected_digest != result_id(value):
                raise EvidencePathError('Feedback evidence changed; restart pagination')
            text = encoded(value)
            interpretation = evidence_path.startswith(('/dispatch_attempts', '/grouping_context/attempts'))
            metadata = {'job_id': job_id, 'evidence_path': evidence_path,
                    'raw_ref': 'journal:' + job_id + root + pointer, 'sha256': result_id(value),
                    'authority': 'untrusted_evidence',
                    'content_authority': 'model_interpretation' if interpretation else 'untrusted_evidence',
                    'encoding': 'json', 'offset': offset, 'total_characters': len(text),
                    'simulated': row['simulated'], 'training_ready': False, 'applied': False}
            chunk = text[offset:offset + limit * 128]

            def page(size):
                end = offset + size
                return {**metadata, 'text': chunk[:size],
                        'next_offset': end if end < len(text) else None}

            complete = page(len(chunk))
            if len(json_bytes(complete)) <= _EVIDENCE_PAGE_BYTES:
                return complete
            # Price the outer JSON string too: quotes, backslashes and control
            # escapes can grow again when JSON evidence is nested inside a page.
            # Keep at least one source character, never a byte fragment or a
            # non-advancing cursor, even if metadata alone exhausts the soft cap.
            lo, hi = min(1, len(chunk)), len(chunk)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(json_bytes(page(mid))) <= _EVIDENCE_PAGE_BYTES:
                    lo = mid
                else:
                    hi = mid - 1
            return page(lo)
        if job_id is not None:
            if expected_digest is not None:
                raise ValueError('expected_digest requires an evidence or knowledge page')
            row = self.state['jobs'][job_id]
            return deepcopy({'job_id': job_id, 'status': row['status'], 'model': row['model'],
                'receipt_digest': row['receipt_digest'], 'proposals': row['proposals'],
                'products': row.get('products', {}),
                'research_context': row['input'].get('research_context', {}),
                'attempt_revisions': deepcopy(row.get('attempt_revisions', [])),
                'grouping_error': row.get('grouping_error'),
                'publication_error': row.get('publication_error'),
                'source_refs': row['input']['source_refs'], 'error': row['error'],
                'evidence_packaging': row['input'].get('evidence_packaging'),
                'context_memory': {'enabled': True,
                                  'read_path': {'operation': 'feedback_learning_read', 'job_id': job_id,
                                                'evidence_path': '/context_memory'}},
                'trajectory': ({'enabled': True,
                               'read_path': {'operation': 'feedback_learning_read', 'job_id': job_id,
                                             'evidence_path': '/trajectory'}}
                               if self.trajectory_archive is not None else {'enabled': False}),
                'simulated': row['simulated'], 'authority': 'unverified_proposals',
                'training_ready': False, 'applied': False})
        rows = self.ordered_jobs()
        if expected_digest is not None:
            raise ValueError('expected_digest requires an evidence or knowledge page')
        if research_attempt_id is not None:
            rows = [row for row in rows if row['input'].get('research_context', {}).get('research_attempt_id') == research_attempt_id
                    or any(r['attempt_id'] == research_attempt_id for r in row.get('attempt_revisions', []))]
        return {'attempted': self.state['attempted'], 'limit': self.limits['max_calls'], 'total': len(rows),
                'items': [{'job_id': r['job_id'], 'status': r['status'],
                           'products': deepcopy(r.get('products', {})),
                           **{key: r['input'].get('research_context', {}).get(key)
                              for key in ('decision_id', 'research_attempt_id', 'research_attempt_title')},
                           'interpreted_attempt_ids': [a['attempt_id'] for a in r.get('attempt_revisions', [])],
                           'grouping_error': r.get('grouping_error'),
                           'kinds': [p['kind'] for p in r['proposals']]} for r in rows[offset:offset+limit]],
                'next_offset': offset+limit if offset+limit < len(rows) else None}
