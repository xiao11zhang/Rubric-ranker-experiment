import contextlib
from copy import deepcopy
import io
import json
import os
import ssl
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, sha
from . import api_extract as pipeline
from . import agent_annotations as agent
from .api_client import ChatAPI, endpoint, load_settings
from .entities import candidate
from .prepare import check_provenance


def rows():
    return [{'id': 'a', 'query': 'PRIVATE QUERY NOT SENT', 'docs': [{'content': 'PRIVATE DOCUMENT'}],
             'rubrics': {'description': 'Alice studied in Paris in 1990; at least 3 sources agree.', 'weight': 5}},
            {'id': 'b', 'query': 'second query', 'docs': [{'content': 'passage'}],
             'rubrics': {'description': 'At least 3 documents agree before 1984.', 'weight': 3}}]


class FakeAPI:
    key = 'test-secret-value-that-must-not-be-saved'
    model = 'test-model'

    def __init__(self, entities=None, decision='approved'):
        self.calls = []
        self.entities = entities if entities is not None else []
        self.decision = decision

    def __call__(self, messages):
        self.calls.append(messages)
        value = {'entities': self.entities}
        return {'content': json.dumps(value), 'returned_model': self.model, 'usage': {'total_tokens': 10}}


class APIExtractionTest(unittest.TestCase):
    def test_truncated_response_is_saved_with_usage_but_never_validated(self):
        class Truncated:
            key = FakeAPI.key
            model = 'test'
            def __call__(self, messages):
                return {'content': '{"entities":[]}', 'finish_reason': 'length',
                        'usage': {'total_tokens': 2048}}
        with self.temporary() as folder:
            record = pipeline.process(rows()[1], Truncated(), 'f', Path(folder)/'record.json')
            self.assertEqual(record['status'], 'error')
            self.assertEqual(record['extraction_response']['usage']['total_tokens'], 2048)
            self.assertIn('finish_reason=length', record['error'])
            self.assertNotIn('candidate', record)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows native trust store')
    def test_native_tls_keeps_certificate_and_hostname_verification(self):
        api = ChatAPI({'RUBRIC_API_KEY': FakeAPI.key,
            'RUBRIC_API_BASE_URL': 'https://example.test/v1', 'RUBRIC_API_MODEL': 'test'},
            {'api_tls_backend': 'windows_truststore'})
        opener = api.verified_opener()
        handler = next(h for h in opener.handlers if hasattr(h, '_context'))
        self.assertEqual(handler._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(handler._context.check_hostname)
        self.assertTrue(api.describe()['certificate_verification'])

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent)

    def test_env_file_quotes_and_environment_precedence(self):
        with self.temporary() as folder:
            path = Path(folder) / '.env'
            path.write_text('RUBRIC_API_KEY="local-secret"\nRUBRIC_API_BASE_URL=https://example.test/v1\nRUBRIC_API_MODEL=model-a\n', encoding='utf-8')
            with patch.dict(os.environ, {'RUBRIC_API_KEY': 'environment-secret', 'RUBRIC_API_BASE_URL': '', 'RUBRIC_API_MODEL': ''}):
                settings = load_settings(path)
                self.assertEqual(settings['RUBRIC_API_KEY'], 'environment-secret')
                self.assertEqual(settings['RUBRIC_API_MODEL'], 'model-a')

    def test_endpoint_and_secret_free_description(self):
        self.assertEqual(endpoint('https://example.test/v1'), 'https://example.test/v1/chat/completions')
        self.assertEqual(endpoint('https://example.test/v1/chat/completions/'), 'https://example.test/v1/chat/completions')
        for bad in ('https://user:password@example.test/v1', 'https://example.test/v1?key=secret', 'file:///tmp/api'):
            with self.assertRaises(ValueError): endpoint(bad)
        api = ChatAPI({'RUBRIC_API_KEY': FakeAPI.key, 'RUBRIC_API_BASE_URL': 'https://example.test/v1', 'RUBRIC_API_MODEL': 'test'}, {})
        self.assertNotIn(FakeAPI.key, json.dumps(api.describe()))

    def test_http_payload_and_redacted_errors(self):
        seen = []
        def open_request(request, timeout):
            seen.append(request)
            body = {'choices': [{'message': {'content': '{"entities":[]}'}, 'finish_reason': 'stop'}], 'model': 'returned-test'}
            return contextlib.closing(io.BytesIO(json.dumps(body).encode()))
        settings = {'RUBRIC_API_KEY': FakeAPI.key, 'RUBRIC_API_BASE_URL': 'https://example.test/v1', 'RUBRIC_API_MODEL': 'test'}
        api = ChatAPI(settings, {'api_json_mode': True, 'api_thinking': 'disabled'}, opener_factory=lambda: SimpleNamespace(open=open_request))
        result = api([{'role': 'user', 'content': 'rubric'}])
        payload = json.loads(seen[0].data)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertNotIn(FakeAPI.key, json.dumps(payload))
        self.assertEqual(result['returned_model'], 'returned-test')
        def forbidden(*a, **kw):
            raise HTTPError('https://example.test', 401, FakeAPI.key, {}, io.BytesIO(FakeAPI.key.encode()))
        api.opener_factory = lambda: SimpleNamespace(open=forbidden)
        with self.assertRaises(RuntimeError) as error: api([])
        self.assertNotIn(FakeAPI.key, str(error.exception))

    def test_api_response_echoing_key_is_not_retained(self):
        settings = {'RUBRIC_API_KEY': FakeAPI.key, 'RUBRIC_API_BASE_URL': 'https://example.test/v1', 'RUBRIC_API_MODEL': 'test'}
        raw = json.dumps({'choices': [{'message': {'content': FakeAPI.key}}]}).encode()
        api = ChatAPI(settings, {}, opener_factory=lambda: SimpleNamespace(open=lambda *a, **kw: contextlib.closing(io.BytesIO(raw))))
        with self.assertRaises(RuntimeError) as error: api([])
        self.assertNotIn(FakeAPI.key, str(error.exception))

    def test_extraction_review_and_preserved_conditions(self):
        with self.temporary() as folder:
            path = Path(folder) / 'record.json'
            api = FakeAPI([{'type': 'person', 'mentions': ['Alice']}, {'type': 'location', 'mentions': ['Paris']}])
            record = pipeline.process(rows()[0], api, 'fingerprint', path)
            self.assertEqual(record['status'], 'validated')
            self.assertEqual(len(api.calls), 1)
            self.assertFalse(record['review']['semantic_reviewed'])
            self.assertNotIn('review_response', record)
            self.assertNotIn('PRIVATE QUERY', json.dumps(api.calls))
            self.assertNotIn('PRIVATE DOCUMENT', json.dumps(api.calls))
            self.assertIn('in 1990; at least 3', record['candidate']['rubrics']['description'])
            self.assertEqual(record['candidate']['rubrics']['weight'], 5)
            self.assertNotIn(api.key, path.read_text(encoding='utf-8'))

    def test_failed_or_rejected_rows_never_approve_or_fall_back(self):
        with self.temporary() as folder:
            path = Path(folder) / 'record.json'
            record = pipeline.process(rows()[0], FakeAPI([{'type': 'person', 'mentions': ['not-in-source']}]), 'f', path)
            self.assertEqual(record['status'], 'error')
            self.assertNotIn('review', record)
            valid = pipeline.process(rows()[1], FakeAPI(), 'f', path)
            self.assertEqual(valid['status'], 'validated')
            self.assertFalse(valid['review']['semantic_reviewed'])

    def test_saved_extraction_resumes_without_any_api_call(self):
        row = rows()[1]
        previous = {'status': 'in_progress', 'extraction_response': {'content': '{"entities":[]}'}}
        class NeverCall:
            key = FakeAPI.key
            model = 'test'
            def __call__(self, messages): raise AssertionError('No API call on saved response')
        with self.temporary() as folder:
            record = pipeline.process(row, NeverCall(), 'f', Path(folder) / 'record.json', previous)
            self.assertEqual(record['status'], 'validated')

    def test_exclusive_run_lock_prevents_duplicate_paid_workers(self):
        with self.temporary() as folder:
            base = Path(folder)
            with pipeline.run_lock(base):
                self.assertTrue((base / 'active.lock').is_file())
                with self.assertRaises(RuntimeError):
                    with pipeline.run_lock(base):
                        self.fail('Second API writer must not start')
            self.assertFalse((base / 'active.lock').exists())

    def test_reuse_agent_batch_does_not_rewrite_old_artifacts(self):
        with self.temporary() as folder:
            root = Path(folder).resolve()
            config, train, baseline, log = [root / x for x in ('config.toml', 'train.jsonl', 'baseline.json', 'log.jsonl')]
            for path in (config, train, baseline, log): path.write_text('{}', encoding='utf-8')
            selected = rows()
            source = ({'train_data': agent.rel(train), 'baseline_plan': agent.rel(baseline)}, {}, selected, log, 'prefix')
            base = root / 'agent'
            with patch.object(agent, 'BASE', base), patch.object(agent, 'sources', return_value=source), contextlib.redirect_stdout(io.StringIO()):
                agent.initialize(config, 2)
                annotations = root / 'annotations.json'
                annotations.write_text(json.dumps({'batch_id': 'batch-0001', 'records': [{'id': 'a', 'entities': [{'type': 'person', 'mentions': ['Alice']}, {'type': 'location', 'mentions': ['Paris']}]}, {'id': 'b', 'entities': []}]}), encoding='utf-8')
                agent.commit('batch-0001', annotations)
                decisions = root / 'decisions.json'
                decisions.write_text(json.dumps({'batch_id': 'batch-0001', 'records': [{'id': x['id'], 'decision': 'approved', 'notes': 'Checked.', 'review_method': agent.IDENTITY['review_method']} for x in selected]}), encoding='utf-8')
                agent.review('batch-0001', decisions)
                before = {str(p): sha(p) for p in base.rglob('*') if p.is_file()}
                imported, artifacts = pipeline.collect_agent(selected)
                self.assertEqual(len(imported), 2)
                self.assertTrue(artifacts)
                self.assertEqual(before, {str(p): sha(p) for p in base.rglob('*') if p.is_file()})

    def test_mixed_finalization_same_directory_and_frozen_provenance(self):
        with self.temporary() as folder:
            root = Path(folder).resolve()
            base = root / 'data/agent/api'
            base.mkdir(parents=True)
            config, train, baseline, log = [root / x for x in ('config.toml', 'train.jsonl', 'baseline.json', 'log.jsonl')]
            for path in (config, train, baseline, log): path.write_text('{}', encoding='utf-8')
            selected = rows()
            c = {'derived_data': str(base.parent / 'candidates.jsonl'), 'review_data': str(base.parent / 'review.jsonl'), 'extraction_manifest': str(base.parent / 'manifest.json')}
            item = candidate(selected[0], [], 'old')
            imported = {'a': {'origin': 'agent_annotations', 'entities': [], 'review': {'id': 'a', 'original_rubric_hash': item['original_rubric_hash'], 'rubric_hash': item['rubric_hash'], 'decision': 'approved', 'reviewer': 'agent fixture', 'review_method': agent.IDENTITY['review_method'], 'human_review_claimed': False}}}
            code = {pipeline.rel(Path(pipeline.__file__)): sha(pipeline.__file__)}
            spec = {'config_sha256': sha(config), 'order_ids': ['a', 'b'], 'train_sha256': sha(train),
                    'baseline_actual_prefix_sha256': 'prefix', 'api': {'requested_model': 'fixture'},
                    'frozen_files_sha256': {pipeline.rel(config): sha(config), **code}, 'code_sha256': code,
                    'imported_agent_sha256': digest(imported), 'rule_version': pipeline.RULE_VERSION, 'audit_policy': pipeline.sampling.POLICY}
            agent.fresh_json(base / 'run.json', {'spec': spec, 'imported_agent_records': imported})
            record = pipeline.process(selected[1], FakeAPI(), digest(spec), base / 'records/0002.json')
            with self.assertRaisesRegex(ValueError, 'audit incomplete'):
                pipeline.sampling.check(base, selected, {'b':record}, imported)
            pipeline.sampling.approve(base, selected, {'b':record}, [{'id':'b', 'rubric_hash':record['candidate']['rubric_hash'], 'decision':'approved', 'notes':'Agent checked no names and unchanged year/quantity.'}])
            with patch.object(agent, 'sources', return_value=(c, {}, selected, log, 'prefix')), patch.object(pipeline, 'REPORTS', root / 'reports'), contextlib.redirect_stdout(io.StringIO()):
                pipeline.finalize(config, base)
            items = [json.loads(x) for x in Path(c['derived_data']).read_text(encoding='utf-8').splitlines()]
            manifest = json.loads(Path(c['extraction_manifest']).read_text(encoding='utf-8'))
            self.assertEqual([x['id'] for x in items], ['a', 'b'])
            self.assertEqual(manifest['source_counts'], {'agent_annotations': 1, 'external_api': 1})
            check_provenance(items, manifest, selected, sha(train))
            self.assertNotIn(FakeAPI.key, json.dumps(manifest))
            (base / 'records/0002.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): check_provenance(items, manifest, selected, sha(train))


if __name__ == '__main__':
    unittest.main()
