"""Render the invoice page when its usual default statuses are unavailable."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auth_support import provision_account, sign_in_app
from recovery.engine import audit
from recovery.storage import connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
try:
    AVAILABLE = (importlib.util.find_spec('streamlit') is not None
                 and importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
                 and importlib.util.find_spec('pydantic') is not None)
except ModuleNotFoundError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Install ui, graph and ollama extras for invoice status-filter verification')
class InvoiceStatusFilterUITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / 'synthetic.sqlite3'
        self.checkpoints = self.db.with_name('synthetic-ui-checkpoints.sqlite3')
        self.auth_db = Path(directory.name) / 'accounts.sqlite3'
        provision_account(self.auth_db)
        load_snapshot(self.db, ROOT / 'data')
        with connect(self.db) as conn:
            self.baseline = audit(conn)
        environment = patch.dict(os.environ, {'RECOVERY_DB': str(self.db),
                                             'RECOVERY_AUTH_DB': str(self.auth_db)})
        environment.start()
        self.addCleanup(environment.stop)
        self.providers = []
        for target in ('recovery.ollama_provider._json_request', 'recovery.ai._model',
                       'recovery.ai.resolve_explanation_config'):
            blocked = patch(target, side_effect=AssertionError('Filtering must not call a model'))
            self.providers.append(blocked.start())
            self.addCleanup(blocked.stop)

    def filtered_report(self, statuses):
        # Exercise the UI report boundary without editing fixture rows or goldens.
        report = deepcopy(self.baseline)
        report['cases'] = [case for case in report['cases'] if case['status'] in statuses]
        report['case_count'] = len(report['cases'])
        report['counts'] = dict(Counter(case['status'] for case in report['cases']))
        report['candidate_total'] = format(sum(
            (Decimal(case['net_candidate']) for case in report['cases']
             if case['status'] == 'CANDIDATE'), Decimal('0.00')), '.2f')
        return report

    def assert_rendered_filter(self, report):
        from streamlit.testing.v1 import AppTest
        from recovery import invoice_review

        with ExitStack() as patches:
            patches.enter_context(patch('recovery.engine.audit', return_value=report))
            patches.enter_context(patch.object(invoice_review, 'audit', return_value=report, create=True))
            app = sign_in_app(AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30),
                              self.auth_db).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(any(item.label == 'Statuses' for item in app.multiselect))
            self.assertTrue(any(item.label == 'Vendor' for item in app.selectbox))
            self.assertTrue(any(item.label == 'Contract' for item in app.selectbox))
            scope = next(item for item in app.radio if item.label == 'Review scope')
            self.assertEqual(scope.value, 'Needs review')
            scope.set_value('All cases').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            frame = next(item.value for item in app.dataframe if 'Open' in item.value.columns)
            line_column = next(name for name in ('Invoice line', 'Case', 'line_id', 'Case ID')
                               if name in frame.columns)
            self.assertEqual(set(frame[line_column]), {case['line_id'] for case in report['cases']})
            app.run()
            self.assertFalse(app.exception)
        for provider in self.providers:
            provider.assert_not_called()
        self.assertFalse(self.checkpoints.exists())
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)
            self.assertEqual(audit(conn), self.baseline)

    def test_invoice_page_renders_without_needs_review_status(self):
        report = self.filtered_report({'CANDIDATE', 'NO_DISCREPANCY', 'ALREADY_OFFSET', 'UNPAID'})
        self.assertNotIn('NEEDS_REVIEW', report['counts'])
        self.assert_rendered_filter(report)

    def test_invoice_page_renders_without_either_default_status(self):
        report = self.filtered_report({'NO_DISCREPANCY', 'ALREADY_OFFSET', 'UNPAID'})
        self.assertNotIn('CANDIDATE', report['counts'])
        self.assertNotIn('NEEDS_REVIEW', report['counts'])
        self.assert_rendered_filter(report)


if __name__ == '__main__':
    unittest.main()
