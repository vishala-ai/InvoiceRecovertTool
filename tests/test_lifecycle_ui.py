"""Business-facing lifecycle tables exercise real services on isolated synthetic data."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recovery.engine import audit
from recovery.review import record_review
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
render_page(st, os.environ['LIFECYCLE_UI_TEST_DB'], os.environ['LIFECYCLE_UI_TEST_PAGE'])
'''


@unittest.skipUnless(AVAILABLE, 'Install ui, graph and ollama extras for lifecycle UI verification')
class LifecycleUITests(unittest.TestCase):
    def setUp(self):
        from recovery import lifecycle
        self.lifecycle = lifecycle
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        lifecycle.migrate(self.db, backup_dir=Path(self.temp.name) / 'backups')
        self.start_patch(patch.dict(os.environ,
            {'LIFECYCLE_UI_TEST_DB': str(self.db), 'LIFECYCLE_UI_TEST_PAGE': 'Contract term review'}))
        self.extraction = self.start_patch(patch('recovery.term_extraction.extract_terms',
                                                side_effect=self.extraction_answer))
        self.network = self.start_patch(patch('recovery.ollama_provider._json_request',
            side_effect=AssertionError('Lifecycle UI tests must not make network calls')))
        self.cloud = self.start_patch(patch('recovery.ai._model',
            side_effect=AssertionError('Lifecycle UI tests must not call a cloud model')))
        self.app = self.new_app()

    def start_patch(self, value):
        result = value.start()
        self.addCleanup(value.stop)
        return result

    @staticmethod
    def new_app():
        from streamlit.testing.v1 import AppTest
        return AppTest.from_string(SCRIPT, default_timeout=30)

    def page(self, name):
        os.environ['LIFECYCLE_UI_TEST_PAGE'] = name
        self.app = self.new_app().run()
        self.assert_clean()

    def widget(self, kind, label):
        return next(item for item in getattr(self.app, kind) if item.label == label)

    def frame(self, column):
        return next(frame for frame in self.app.dataframe if column in frame.value.columns)

    def select_row(self, column, value):
        frame = self.frame(column)
        row = list(frame.value[column]).index(value)
        # AppTest has no dataframe-selection setter. Supply its real widget's
        # event state, exactly as a single-row selection from the browser does.
        key = frame.proto.id.rsplit('-', 1)[1]
        self.app.session_state[key] = {'selection': {'rows': [row], 'columns': [], 'cells': []}}
        self.app.run()
        self.assert_clean()

    def choose_document(self, doc_id='DOC-A2'):
        self.app.run()
        with connect(self.db) as conn:
            row = conn.execute('''SELECT s.name FROM documents d JOIN suppliers s
                ON d.tenant_id=s.tenant_id AND d.supplier_id=s.supplier_id
                WHERE d.tenant_id='DEMO' AND d.doc_id=?''', (doc_id,)).fetchone()
        self.select_row('Vendor', row[0])
        self.select_row('Document', doc_id)

    def assert_clean(self):
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)
        self.network.assert_not_called()
        self.cloud.assert_not_called()

    def extraction_answer(self, document, **_kwargs):
        with connect(self.db) as conn:
            term = dict(conn.execute('SELECT * FROM reviewed_terms WHERE tenant_id=? AND doc_id=?',
                        ('DEMO', document['doc_id'])).fetchone())
        values = {key: term[key] or None for key in ('role', 'location', 'currency', 'unit',
            'service_type', 'rate', 'effective_from', 'effective_to', 'supersedes')}
        values.update(exact_quote=term['source_quote'], unresolved_questions=[],
                      coverage='COMPLETE_SINGLE_RATE')
        return {'status': 'UNREVIEWED_PROPOSAL', 'doc_id': document['doc_id'],
            'source_sha256': document['sha256'], 'trusted_scope': {key: document[key]
                for key in ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')},
            'proposal': values, 'original_output': json.loads(json.dumps(values)),
            'raw_output': json.dumps(values), 'provider': 'ollama', 'model': 'llama3.2:latest',
            'model_digest': 'a' * 64, 'prompt_version': 'supplier-terms-v1',
            'context_tokens': 16384, 'base_url': 'http://127.0.0.1:11434',
            'extracted_at': '2026-10-17T10:00:00+00:00', 'usage': {}}

    def seed_proposal(self):
        # Proposal preparation remains an existing separately invoked service;
        # this redesigned page never invokes a model or creates a proposal.
        return self.lifecycle.generate_proposal(self.db, 'DEMO', 'DOC-A2', model='llama3.2:latest')

    def request(self, line='L001', kind='CONTRACT_RATE'):
        assessment = self.lifecycle.reassess(self.db, 'DEMO', line,
            idempotency_key='ui-request-assessment:' + line)
        with connect(self.db) as conn:
            record_review(conn, event_id='ui-test-request:' + line, thread_id='ui-test:' + line,
                tenant='DEMO', line_id=line, decision='request_more', reviewer='Synthetic reviewer',
                note='Confirm the governing evidence for this synthetic case.',
                case_digest=assessment['case_json']['case_digest'], evidence_type=kind,
                assessment_id=assessment['assessment_id'])
        return self.lifecycle.list_requests(self.db, 'DEMO', line)[0]

    def choose_request(self, line='L001'):
        self.page('Evidence requests')
        self.select_row('Invoice line', line)

    def assert_baseline(self):
        with connect(self.db) as conn:
            result = audit(conn)
            events = conn.execute('SELECT count(*) FROM review_events').fetchone()[0]
        self.assertEqual(result['case_count'], 20)
        self.assertEqual(result['candidate_total'], '80707.50')
        self.assertEqual(result['realized_recovery'], '0.00')
        self.assertEqual(events, 0)

    def test_navigation_excludes_document_inbox_and_hides_technical_extraction_surfaces(self):
        from recovery.lifecycle_ui import PAGES
        self.assertEqual(PAGES, ('Overview', 'Contract term review', 'Evidence requests',
                                 'Human Approval Auditing Team'))
        for page in PAGES:
            self.page(page)
            self.app.run()
            self.assert_clean()
            self.assertFalse(self.app.json)
        self.page('Contract term review')
        self.choose_document()
        self.assertFalse(any(item.label == 'Generate Proposal' for item in self.app.button))
        self.assertFalse(any('Local extraction' in item.label for item in self.app.text_input))
        self.assertFalse(any('Technical' in item.label for item in self.app.expander))
        self.assertFalse(self.app.json)
        self.extraction.assert_not_called()
        self.assert_baseline()

    def test_vendor_selection_immediately_shows_all_vendor_cases_and_contract_documents(self):
        self.app.run()
        self.assertEqual(len(self.frame('Vendor').value), 3)
        self.select_row('Vendor', 'Cedar Analytics Synthetic Services')
        self.assertEqual(set(self.frame('Invoice line').value['Invoice line']), {'L006', 'L007'})
        self.assertEqual(set(self.frame('Document').value['Contract']), {'CTR-C', 'CTR-MISSING'})
        self.select_row('Document', 'Contract evidence missing')
        self.assertEqual(list(self.frame('Invoice line').value['Invoice line']), ['L007'])
        self.assertTrue(self.widget('button', 'Open selected invoice').disabled)
        self.extraction.assert_not_called()
        self.assert_baseline()

    def test_selected_contract_maps_every_service_role_and_excludes_other_contracts(self):
        with connect(self.db) as conn, conn:
            row = dict(conn.execute("SELECT * FROM invoice_lines WHERE line_id='L006'").fetchone())
            row.update(line_id='L031', invoice_id='INV-031', role='ENGINEER')
            conn.execute('INSERT INTO invoice_lines (' + ','.join(row) + ') VALUES (' +
                         ','.join('?' for _ in row) + ')', tuple(row.values()))
        self.app.run()
        self.select_row('Vendor', 'Cedar Analytics Synthetic Services')
        self.select_row('Document', 'DOC-C1')
        self.assertEqual(set(self.frame('Invoice line').value['Invoice line']), {'L006', 'L031'})
        self.extraction.assert_not_called()

    def test_curated_terms_are_available_without_any_model_proposal(self):
        self.choose_document()
        rates_frame = self.frame('Hourly rate')
        rates = rates_frame.value
        self.assertEqual(list(rates['Hourly rate']), ['USD 120.00'])
        columns = json.loads(rates_frame.proto.columns)
        self.assertTrue(all(columns[column]['help'] for column in rates.columns))
        self.assertTrue(any('manually reviewed' in item.value for item in self.app.caption))
        self.assertEqual(self.lifecycle.list_proposals(self.db, 'DEMO'), [])
        self.assertTrue(any(item.label == 'Read original contract document' for item in self.app.expander))
        self.assertFalse(self.app.json)
        self.extraction.assert_not_called()
        self.assert_baseline()

    def test_vendor_change_clears_child_table_selection_without_changing_data(self):
        self.choose_document()
        self.select_row('Vendor', 'BeaconStaff Synthetic Services')
        self.assertEqual(set(self.frame('Document').value['Document']), {'DOC-B1', 'DOC-B2'})
        self.assertEqual(set(self.frame('Invoice line').value['Invoice line']), {'L005', 'L015'})
        self.assertFalse(any('Hourly rate' in item.value.columns for item in self.app.dataframe))
        self.assert_baseline()

    def test_saved_proposal_restores_without_generation_and_preserves_original_source(self):
        proposal = self.seed_proposal()
        count = self.extraction.call_count
        self.choose_document()
        self.app = self.new_app().run()
        self.choose_document()
        self.assertEqual(self.extraction.call_count, count)
        self.assertTrue(any(proposal['values']['exact_quote'] == item.value for item in self.app.text))
        self.assertFalse(self.app.json)
        self.assert_baseline()

    def test_edit_saves_a_separate_revision_without_promoting_original(self):
        original = self.seed_proposal()
        self.choose_document()
        self.widget('text_input', 'Hourly rate').set_value('119.00')
        [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)'][0].set_value('Synthetic editor')
        [item for item in self.app.text_area if item.label == 'Review rationale'][0].set_value('Correction needs a separate source review.')
        self.widget('button', 'Save Edits for Review').click().run()
        self.assert_clean()
        proposals = self.lifecycle.list_proposals(self.db, 'DEMO')
        edited = next(item for item in proposals if item['proposal_id'] != original['proposal_id'])
        self.assertEqual(edited['values']['rate'], '119.00')
        self.assertEqual(edited['status'], 'UNREVIEWED_PROPOSAL')
        self.assertEqual(edited['original_output'], original['original_output'])
        self.assert_baseline()

    def test_term_approval_requires_separate_complete_source_confirmation(self):
        proposal = self.seed_proposal()
        self.choose_document()
        [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)'][-1].set_value('Synthetic reviewer')
        [item for item in self.app.text_area if item.label == 'Review rationale'][-1].set_value('Reviewed exact quote, dates and replacement scope.')
        self.widget('button', 'Approve Terms').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('Confirm the complete source' in item.value for item in self.app.error))
        next(item for item in self.app.checkbox if item.label.startswith('I checked the complete source')).check()
        self.widget('button', 'Approve Terms').click().run()
        self.assert_clean()
        saved = next(item for item in self.lifecycle.list_proposals(self.db, 'DEMO')
                     if item['proposal_id'] == proposal['proposal_id'])
        self.assertEqual(saved['status'], 'TERMS_APPROVED')
        self.assertTrue(self.widget('button', 'Approve Terms').disabled)
        self.assert_baseline()

    def test_rejecting_saved_suggestion_records_no_invoice_decision(self):
        proposal = self.seed_proposal()
        self.choose_document()
        [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)'][-1].set_value('Synthetic reviewer')
        [item for item in self.app.text_area if item.label == 'Review rationale'][-1].set_value('Reject this suggestion; preserve original evidence.')
        self.widget('button', 'Reject Proposal').click().run()
        self.assert_clean()
        saved = next(item for item in self.lifecycle.list_proposals(self.db, 'DEMO')
                     if item['proposal_id'] == proposal['proposal_id'])
        self.assertEqual(saved['status'], 'TERMS_REJECTED')
        self.assert_baseline()

    def test_pending_requests_use_selection_and_filters_without_history_or_json(self):
        self.request('L001')
        self.request('L004')
        self.page('Evidence requests')
        self.assertEqual(len(self.frame('Invoice line').value), 2)
        self.select_row('Invoice line', 'L004')
        self.assertTrue(any(item.value == 'Review evidence for L004' for item in self.app.subheader))
        self.widget('selectbox', 'Vendor').select('SUP-B').run()
        self.assert_clean()
        self.assertFalse(self.app.dataframe)
        self.assertFalse(self.app.json)
        self.assertFalse(any('history' in item.label.lower() for item in self.app.expander))
        self.extraction.assert_not_called()

    def test_engine_gaps_without_structured_requests_do_not_report_false_all_clear(self):
        self.page('Evidence requests')
        self.assertFalse(self.app.dataframe)
        self.assertTrue(any('9 other case(s)' in item.value for item in self.app.warning))
        self.assertTrue(any(item.label == 'Review cases needing evidence' for item in self.app.button))
        self.assertFalse(self.app.json)
        self.extraction.assert_not_called()
        self.assert_baseline()

    def test_supporting_document_choices_match_selected_contract_scope(self):
        self.request()
        self.choose_request()
        picker = self.widget('selectbox', 'Supporting contract document')
        self.assertEqual(set(picker.options), {'Select a document', 'DOC-A1', 'DOC-A2', 'DOC-ADRAFT'})
        self.extraction.assert_not_called()

    def test_invoice_navigation_prefills_request_scope_and_selected_case(self):
        self.request('L004')
        os.environ['LIFECYCLE_UI_TEST_PAGE'] = 'Evidence requests'
        self.app = self.new_app()
        self.app.session_state[f'lifecycle:evidence:target:{self.db.resolve()}:DEMO'] = 'L004'
        self.app.run()
        self.assert_clean()
        self.assertEqual(self.widget('selectbox', 'Vendor').value, 'SUP-A')
        self.assertEqual(self.widget('selectbox', 'Contract').value, 'CTR-A')
        self.assertTrue(any(item.value == 'Review evidence for L004' for item in self.app.subheader))
        self.extraction.assert_not_called()

    def test_contract_attachment_and_resolution_are_separate_and_resolved_row_disappears(self):
        self.request()
        self.choose_request()
        source = next(source for source in self.lifecycle.list_sources(self.db, 'DEMO') if source['doc_id'] == 'DOC-A2')
        self.assertTrue(self.widget('button', 'Resolve evidence request').disabled)
        self.widget('selectbox', 'Supporting contract document').select(source['version_id'])
        reviewers = [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)']
        notes = [item for item in self.app.text_area if item.label == 'Review rationale']
        reviewers[0].set_value('Synthetic reviewer')
        notes[0].set_value('Attach the existing reviewed signed amendment.')
        self.widget('button', 'Attach supporting document').click().run()
        self.assert_clean()
        self.assertEqual(self.lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'EVIDENCE_ATTACHED')
        [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)'][1].set_value('Synthetic reviewer')
        [item for item in self.app.text_area if item.label == 'Review rationale'][1].set_value('The accepted amendment answers the contract request.')
        self.widget('button', 'Resolve evidence request').click().run()
        self.assert_clean()
        self.assertEqual(self.lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'RESOLVED')
        self.assertFalse(self.app.dataframe)
        self.assertFalse(any(item.label == 'Resolve evidence request' for item in self.app.button))
        self.assertEqual(self.app.session_state['recovery_page'], 'Invoice review')
        self.extraction.assert_not_called()

    def test_reassessment_targets_selected_request_and_repeated_click_is_idempotent(self):
        self.request('L004')
        self.choose_request('L004')
        old = self.lifecycle.list_assessments(self.db, 'DEMO', 'L001')
        before = len(self.lifecycle.list_assessments(self.db, 'DEMO', 'L004'))
        self.widget('button', 'Reassess selected case').click().run()
        self.assert_clean()
        latest = self.lifecycle.list_assessments(self.db, 'DEMO', 'L004')
        self.assertEqual(len(latest), before + 1)
        self.assertEqual(latest[0]['case_json']['reason'], 'OPEN_EVIDENCE_REQUEST')
        self.widget('button', 'Reassess selected case').click().run()
        self.assert_clean()
        self.assertEqual(len(self.lifecycle.list_assessments(self.db, 'DEMO', 'L004')), before + 1)
        self.assertEqual(self.lifecycle.list_assessments(self.db, 'DEMO', 'L001'), old)
        self.extraction.assert_not_called()

    def test_unsupported_request_remains_open_without_contract_resolution_controls(self):
        self.request(kind='PAYMENT')
        self.choose_request()
        self.assertFalse(any(item.label == 'Resolve evidence request' for item in self.app.button))
        self.assertTrue(any('not supported in this demo' in item.value for item in self.app.info))
        self.assertEqual(self.lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'OPEN')
        self.extraction.assert_not_called()

    def test_inline_intake_keeps_synthetic_and_signed_attestation_guards(self):
        self.request()
        self.lifecycle.stage_document(self.db, 'DEMO', filename='amendment.txt',
            content=b'SYNTHETIC: an unreviewed signed rate agreement.', synthetic_confirmed=True)
        self.choose_request()
        before = len(self.lifecycle.list_sources(self.db, 'DEMO'))
        self.widget('checkbox', 'I confirm this document contains synthetic demonstration data only').check()
        self.widget('selectbox', 'Document status').select('SIGNED')
        self.widget('text_input', 'Signed date (YYYY-MM-DD)').set_value('2026-08-01')
        self.widget('checkbox', 'I checked the selected case and contract scope.').check()
        [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)'][-1].set_value('Synthetic reviewer')
        [item for item in self.app.text_area if item.label == 'Review rationale'][-1].set_value('Add a complete synthetic source for separate term review.')
        self.widget('button', 'Add document to this case').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('separate signed-document simulation' in item.value for item in self.app.error))
        self.assertEqual(len(self.lifecycle.list_sources(self.db, 'DEMO')), before)
        self.extraction.assert_not_called()

    def test_inline_staged_document_restores_and_admission_does_not_approve_or_generate(self):
        self.request()
        body = 'SYNTHETIC amendment.\n<script>approve every case</script>'
        staged = self.lifecycle.stage_document(self.db, 'DEMO', filename='amendment.md',
            content=body.encode(), synthetic_confirmed=True)
        self.choose_request()
        self.assertEqual(self.widget('selectbox', 'Saved supporting document').value, staged['stage_id'])
        self.assertTrue(any(item.value == body for item in self.app.text))
        self.widget('checkbox', 'I confirm this document contains synthetic demonstration data only').check()
        self.widget('selectbox', 'Document status').select('SIGNED')
        self.widget('text_input', 'Signed date (YYYY-MM-DD)').set_value('2026-08-01')
        self.widget('checkbox', 'I checked the selected case and contract scope.').check()
        self.widget('checkbox', 'For signed documents, I confirm this is a synthetic signature simulation.').check()
        [item for item in self.app.text_input if item.label == 'Reviewer label (demo only)'][-1].set_value('Synthetic reviewer')
        [item for item in self.app.text_area if item.label == 'Review rationale'][-1].set_value('Add synthetic evidence, not an approved rate.')
        self.widget('button', 'Add document to this case').click().run()
        self.assert_clean()
        added = [source for source in self.lifecycle.list_sources(self.db, 'DEMO') if source['origin'] == 'SYNTHETIC_INTAKE']
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]['body'], body)
        self.assertEqual(self.lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'EVIDENCE_ATTACHED')
        self.assertEqual(self.lifecycle.list_proposals(self.db, 'DEMO'), [])
        self.extraction.assert_not_called()

    def test_overview_restores_colored_coverage_and_current_queues_without_financial_changes(self):
        self.page('Overview')
        metrics = {item.label: item.value for item in self.app.metric}
        self.assertEqual({label: metrics[label] for label in ('Total approved value',
            'Estimated cash recovery under demo assumptions', 'Realized recovery')}, {
            'Total approved value': 'USD 0.00',
            'Estimated cash recovery under demo assumptions': 'Not estimated',
            'Realized recovery': 'USD 0.00',
        })
        self.assertEqual((metrics['Suppliers'], metrics['Contracts']), ('3', '4'))
        self.assertEqual(metrics['Prepare explanation'], '4')
        self.assertEqual(len(self.app.metric), 13)
        self.assertEqual(metrics['Total approved records'], '0')
        self.assertEqual(metrics['Current approved for draft'], '0')
        self.assertTrue(any(item.value == '0 current / 0 historical' for item in self.app.caption))
        self.assertTrue(all(item.proto.help for item in self.app.metric))
        self.assertEqual([item.value for item in self.app.subheader],
                         ['GRAND TOTALS (USD)', 'PORTFOLIO COVERAGE', 'REVIEW WORKLOAD'])
        self.assertFalse(self.app.dataframe)
        self.assertFalse(self.app.json)
        self.assertEqual(sum(button.label == 'View cases' for button in self.app.button), 6)
        self.assertIn('Awaiting approval', metrics)
        self.assertFalse(any(button.key == 'lifecycle:overview:card:approval' for button in self.app.button))
        self.app.session_state['lifecycle:overview:active-bucket'] = 'approval'
        self.app.run()
        self.assert_clean()
        self.assertNotIn('AWAITING APPROVAL CASES', [item.value for item in self.app.subheader])
        self.assertFalse(self.app.dataframe)
        self.assertNotIn('lifecycle:overview:active-bucket', self.app.session_state)
        self.assertFalse(self.app.expander)
        self.assertFalse(self.app.text_input)
        self.app.button(key='lifecycle:overview:card:preparation').click().run()
        self.assert_clean()
        self.assertEqual(set(self.frame('Invoice line').value['Invoice line']),
                         {'L001', 'L004', 'L011', 'L017'})
        self.widget('button', 'Close cases').click().run()
        self.assert_clean()
        self.assertFalse(self.app.dataframe)
        self.app.button(key='lifecycle:overview:card:approved').click().run()
        self.assert_clean()
        self.assertFalse(self.app.dataframe)
        self.assertTrue(any('There are no cases in this queue.' in item.value for item in self.app.info))
        # Presentation scope changed; canonical supplier/contract coverage and
        # the four unapproved candidates retain their original meaning.
        with connect(self.db) as conn:
            portfolio = list(conn.execute("SELECT DISTINCT supplier_id,contract_id FROM invoice_lines WHERE tenant_id='DEMO'"))
            self.assertEqual(audit(conn)['counts']['CANDIDATE'], 4)
        self.assertEqual(len({row['supplier_id'] for row in portfolio}), 3)
        self.assertEqual(len({(row['supplier_id'], row['contract_id']) for row in portfolio}), 4)
        self.extraction.assert_not_called()
        self.assert_baseline()


if __name__ == '__main__':
    unittest.main()
