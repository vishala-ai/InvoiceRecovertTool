"""Keep evidence resolution and reassessment on the same synthetic invoice."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auth_support import provision_account, sign_in_app
from recovery import lifecycle
from recovery.engine import audit
from recovery.storage import connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
try:
    AVAILABLE = (importlib.util.find_spec('streamlit') is not None
                 and importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
                 and importlib.util.find_spec('pydantic') is not None)
except ModuleNotFoundError:
    AVAILABLE = False

SCRIPT = '''
import os
import streamlit as st
from recovery.lifecycle_ui import render_page
render_page(st, os.environ['REQUEST_RESOLUTION_TEST_DB'], 'Evidence requests')
'''


@unittest.skipUnless(AVAILABLE, 'Install ui, graph and ollama extras for request-resolution UI verification')
class RequestResolutionUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Path(self.directory.name) / 'synthetic.sqlite3'
        self.checkpoints = Path(self.directory.name) / 'synthetic-ui-checkpoints.sqlite3'
        self.auth_db = Path(self.directory.name) / 'accounts.sqlite3'
        provision_account(self.auth_db)
        load_snapshot(self.db, ROOT / 'data')
        lifecycle.migrate(self.db, backup_dir=Path(self.directory.name) / 'backups')
        environment = patch.dict(os.environ, {'REQUEST_RESOLUTION_TEST_DB': str(self.db),
            'RECOVERY_DB': str(self.db), 'RECOVERY_AUTH_DB': str(self.auth_db)})
        environment.start()
        self.addCleanup(environment.stop)
        self.providers = []
        for target in ('recovery.ollama_provider._json_request', 'recovery.ai._model',
                       'recovery.term_extraction.extract_terms'):
            blocked = patch(target, side_effect=AssertionError('This UI path must not call a model'))
            self.providers.append(blocked.start())
            self.addCleanup(blocked.stop)

    def request(self, line_id):
        from recovery.workflow import run_persistent

        thread_id = 'synthetic-request-' + line_id
        started = run_persistent(self.db, self.checkpoints, thread_id, line_id=line_id)
        self.assertTrue(started['paused'])
        completed = run_persistent(self.db, self.checkpoints, thread_id, decision={
            'decision': 'request_more', 'reviewer': 'Synthetic reviewer',
            'note': 'Confirm the governing signed contract amendment.'})
        self.assertFalse(completed['paused'])
        return lifecycle.list_requests(self.db, 'DEMO', line_id)[0]

    def open_page(self):
        from streamlit.testing.v1 import AppTest

        self.app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        self.assert_clean()

    def widget(self, kind, label):
        return next(item for item in getattr(self.app, kind) if item.label == label)

    def assert_clean(self):
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)
        for provider in self.providers:
            provider.assert_not_called()

    def select_request(self, request):
        frame = next(item for item in self.app.dataframe if 'Evidence needed' in item.value.columns)
        row = next(index for index, line in enumerate(frame.value['Invoice line'])
                   if line == request['line_id'])
        key = frame.proto.id.rsplit('-', 1)[1]
        self.app.session_state[key] = {'selection': {'rows': [row], 'columns': [], 'cells': []}}
        self.app.run()
        self.assert_clean()
        self.assertTrue(any('Review evidence for ' + request['line_id'] == item.value
                            for item in self.app.subheader))

    def test_selected_request_reassesses_its_invoice_and_links_prior_decision(self):
        request = self.request('L004')
        self.open_page()
        self.select_request(request)
        self.assertFalse(any(item.label == 'Invoice line to reassess' for item in self.app.selectbox))
        other_history = lifecycle.list_assessments(self.db, 'DEMO', 'L001')
        prior_count = len(lifecycle.list_assessments(self.db, 'DEMO', 'L004'))
        self.widget('button', 'Reassess selected case').click().run()
        self.assert_clean()
        history = lifecycle.list_assessments(self.db, 'DEMO', 'L004')
        self.assertEqual(len(history), prior_count + 1)
        self.assertEqual(history[0]['prior_review_event_id'], request['review_event_id'])
        self.assertEqual(history[0]['case_json']['reason'], 'OPEN_EVIDENCE_REQUEST')
        self.assertEqual(lifecycle.list_assessments(self.db, 'DEMO', 'L001'), other_history)

    def test_changing_request_rows_changes_reassessment_invoice_and_lineage(self):
        first = self.request('L001')
        second = self.request('L004')
        self.open_page()
        for selected in (second, first, second):
            with self.subTest(line=selected['line_id']):
                self.select_request(selected)
                before = len(lifecycle.list_assessments(self.db, 'DEMO', selected['line_id']))
                self.widget('button', 'Reassess selected case').click().run()
                self.assert_clean()
                history = lifecycle.list_assessments(self.db, 'DEMO', selected['line_id'])
                self.assertGreaterEqual(len(history), before)
                self.assertEqual(history[0]['prior_review_event_id'], selected['review_event_id'])
                self.assertEqual(history[0]['case_json']['line_id'], selected['line_id'])
        self.assertFalse(any(item.label in {'Invoice line to reassess',
            'Link to prior evidence decision (optional)'} for item in self.app.selectbox))

    def test_resolved_request_leaves_pending_queue_and_reassessment_creates_no_approval(self):
        from streamlit.testing.v1 import AppTest
        from recovery.workflow import get_review_workflow

        request = self.request('L004')
        original_workflow = get_review_workflow(self.db, self.checkpoints, 'synthetic-request-L004')
        self.open_page()
        self.select_request(request)
        source = next(source for source in lifecycle.list_sources(self.db, 'DEMO')
                      if source['doc_id'] == 'DOC-A2')
        self.widget('selectbox', 'Supporting contract document').select(source['version_id'])
        prefix = f'lifecycle:evidence:{self.db.resolve()}:DEMO:attachment:' + request['request_id']
        self.app.text_input(key=prefix + ':reviewer').set_value('Synthetic reviewer')
        self.app.text_area(key=prefix + ':note').set_value('Attach the existing reviewed signed amendment.')
        self.widget('button', 'Attach supporting document').click().run()
        self.assert_clean()
        self.assertEqual(lifecycle.list_requests(self.db, 'DEMO', 'L004')[0]['status'], 'EVIDENCE_ATTACHED')
        prefix = f'lifecycle:evidence:{self.db.resolve()}:DEMO:resolution:' + request['request_id']
        self.app.text_input(key=prefix + ':reviewer').set_value('Synthetic reviewer')
        self.app.text_area(key=prefix + ':note').set_value('The reviewed amendment satisfies this contract request.')
        self.widget('button', 'Resolve evidence request').click().run()
        self.assert_clean()
        self.assertFalse(any(item.label == 'Resolve evidence request' for item in self.app.button))
        self.assertFalse(any('Evidence needed' in item.value.columns for item in self.app.dataframe))
        retained = lifecycle.list_requests(self.db, 'DEMO', 'L004')
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['status'], 'RESOLVED')

        self.app = sign_in_app(AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30),
                               self.auth_db)
        self.app.session_state['recovery_page'] = 'Invoice review'
        self.app.session_state[f'invoice_selection:{self.db.resolve()}'] = 'L004'
        self.app.run()
        self.assert_clean()
        reassess_button = next(item for item in self.app.button
                               if item.label.lower() == 'reassess with current evidence')
        reassess_button.click().run()
        self.assert_clean()
        history = lifecycle.list_assessments(self.db, 'DEMO', 'L004')
        self.assertEqual(history[0]['validity'], 'CURRENT')
        self.assertEqual(history[0]['case_json']['status'], 'CANDIDATE')
        self.assertEqual(history[0]['case_json']['net_candidate'], '600.00')
        self.assertEqual(history[0]['prior_review_event_id'], request['review_event_id'])
        self.assertEqual(get_review_workflow(self.db, self.checkpoints, 'synthetic-request-L004'), original_workflow)
        with connect(self.db) as conn:
            events = [dict(row) for row in conn.execute('SELECT * FROM review_events')]
            report = audit(conn)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['decision'], 'NEEDS_MORE_EVIDENCE')
        self.assertEqual(report['candidate_total'], '80707.50')
        self.assertEqual(report['realized_recovery'], '0.00')


if __name__ == '__main__':
    unittest.main()
