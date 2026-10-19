"""Case counts and financial reporting are current-version, currency-safe facts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from recovery.ai import PROMPT_VERSION
from recovery.dashboard import summarize_cases
from recovery.engine import audit
from recovery.retrieval import deterministic_brief
from recovery.storage import connect, load_snapshot


def case(line='L001', amount='800.00', *, tenant='DEMO', digest='v1',
         status='CANDIDATE', currency='USD'):
    return {'tenant_id': tenant, 'line_id': line, 'currency': currency,
            'case_digest': digest, 'status': status, 'net_candidate': amount,
            'reason': 'TEST_ONLY', 'citations': []}


def event(case, decision='APPROVED_FOR_DRAFT', *, thread='ui:00000000000000001000:first',
          recorded_at='2026-10-17T12:00:00+00:00', event_id='review-1'):
    return {'tenant_id': case['tenant_id'], 'line_id': case['line_id'],
            'case_digest': case['case_digest'], 'thread_id': thread,
            'event_id': event_id, 'decision': decision, 'recorded_at': recorded_at,
            'reviewer': 'Demo reviewer', 'note': 'Synthetic test decision.'}


def workflow(case, *, thread='ui:00000000000000001000:first', paused=True,
             brief=True, use_llm=False):
    state = {'tenant': case['tenant_id'], 'line_id': case['line_id'], 'case': case,
             'use_llm': use_llm}
    if brief:
        state['brief'] = deterministic_brief(case)
    return {'thread_id': thread, 'paused': paused,
            'next': ['approval'] if paused else [], 'state': state}


def assessment(case, sequence=1, validity='CURRENT'):
    return {'tenant_id': case['tenant_id'], 'line_id': case['line_id'],
            'assessment_id': f'assessment-{sequence}', 'sequence': sequence,
            'case_digest': case['case_digest'], 'input_digest': f'input-{sequence}',
            'validity': validity, 'case_json': json.dumps(case)}


class DashboardTests(unittest.TestCase):
    def test_original_fixture_baseline_is_computed_without_expected_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'synthetic.sqlite3'
            load_snapshot(db, Path(__file__).resolve().parents[1] / 'data')
            with connect(db) as conn:
                cases = audit(conn)['cases']
        result = summarize_cases(cases, events=[])
        self.assertEqual(result['case_count'], 20)
        self.assertEqual(result['counts'], dict(approved=0, rejected=0, evidence=9,
            approval=0, preparation=4, reassessment=0, other=7))
        self.assertEqual(result['totals']['current_candidate'], '80707.50')
        self.assertEqual(result['totals']['realized_recovery'], '0.00')
        self.assertIsNone(result['totals']['expected_cash'])
        self.assertEqual(result['buckets']['evidence']['unknown_count'], 9)

    def test_review_repeats_count_one_case_and_latest_decision(self):
        original = case()
        events = [event(original, event_id='first'),
                  event(original, 'REJECTED', recorded_at='2026-10-17T13:00:00+00:00',
                        event_id='second'),
                  event(original, 'NEEDS_MORE_EVIDENCE',
                        recorded_at='2026-10-17T14:00:00+00:00', event_id='third')]
        result = summarize_cases([original], events=events)
        self.assertEqual(result['case_count'], 1)
        self.assertEqual(result['counts']['evidence'], 1)
        self.assertEqual(result['cases'][0]['current_decision'], 'NEEDS_MORE_EVIDENCE')
        self.assertEqual(result['totals']['current_candidate'], '800.00')
        self.assertEqual(result['totals']['held_candidate'], '800.00')

    def test_approval_is_subset_and_rejection_does_not_rewrite_calculation(self):
        first, second = case(), case('L002', '500.00')
        result = summarize_cases([first, second], events=[event(first), event(second, 'REJECTED')])
        self.assertEqual(result['counts']['approved'], 1)
        self.assertEqual(result['counts']['rejected'], 1)
        self.assertEqual(result['totals']['current_candidate'], '1300.00')
        self.assertEqual(result['totals']['approved_draft'], '800.00')
        self.assertEqual(result['totals']['rejected_candidate'], '500.00')
        self.assertEqual(result['totals']['realized_recovery'], '0.00')

    def test_old_digest_approval_never_approves_new_amount(self):
        old, new = case(), case(amount='1300.00', digest='v2')
        result = summarize_cases([new], events=[event(old)])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['totals']['approved_draft'], '0.00')
        self.assertIsNone(result['cases'][0]['current_decision'])

    def test_stale_latest_workflow_suppresses_old_approval_and_current_value(self):
        old, new = case(), case(amount='1300.00', digest='v2')
        result = summarize_cases([new], events=[event(old)], workflows=[workflow(old)])
        self.assertEqual(result['counts']['reassessment'], 1)
        self.assertEqual(result['totals']['current_candidate'], '0.00')
        self.assertEqual(result['totals']['historical_stale_candidate'], '800.00')
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_new_review_attempt_suspends_prior_approval(self):
        current = case()
        old = workflow(current, paused=False)
        new = workflow(current, thread='ui:00000000000000002000:second')
        result = summarize_cases([current], events=[event(current)], workflows=[new, old])
        self.assertEqual(result['counts']['approval'], 1)
        self.assertEqual(result['counts']['approved'], 0)
        self.assertIsNone(result['cases'][0]['current_decision'])
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_failed_generation_is_preparation_not_awaiting_approval(self):
        current = case()
        old = workflow(current, paused=False)
        failed = workflow(current, thread='ui:00000000000000002000:second', brief=False)
        failed['next'] = ['draft']
        result = summarize_cases([current], events=[event(current)], workflows=[old, failed])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['counts']['approved'], 0)
        self.assertEqual(result['counts']['approval'], 0)

    def test_model_brief_must_pass_every_validation_gate(self):
        current = case()
        saved = workflow(current, use_llm=True)
        saved['state']['explanation_config'] = {'prompt_version': PROMPT_VERSION}
        saved['state']['brief'] = {'composition': 'validated_fact_template',
            'case_digest': current['case_digest'], 'validation': {'status': 'PASSED',
            'financial_consistency': True, 'citation_check': 'PASSED', 'action_check': 'PASSED'}}
        result = summarize_cases([current], events=[], workflows=[saved])
        self.assertEqual(result['counts']['approval'], 1)
        for field, value in [('status', 'WITHHELD'), ('financial_consistency', False),
                             ('citation_check', 'WITHHELD'), ('action_check', 'WITHHELD')]:
            with self.subTest(field=field):
                invalid = deepcopy(saved)
                invalid['state']['brief']['validation'][field] = value
                result = summarize_cases([current], events=[], workflows=[invalid])
                self.assertEqual(result['counts']['preparation'], 1)

    def test_unvalidated_arbitrary_json_is_not_human_review_pause(self):
        current = case()
        saved = workflow(current, use_llm=True)
        saved['state']['brief'] = {'summary': 'Looks valid', 'citations': []}
        result = summarize_cases([current], events=[], workflows=[saved])
        self.assertEqual(result['counts']['preparation'], 1)

    def test_journal_commit_wins_during_interrupted_checkpoint_completion(self):
        current = case()
        result = summarize_cases([current], events=[event(current)], workflows=[workflow(current)])
        self.assertEqual(result['counts']['approved'], 1)
        self.assertEqual(result['counts']['approval'], 0)

    def test_checkpoint_outcome_without_journal_is_not_an_approval(self):
        current = case()
        saved = workflow(current, paused=False)
        saved['state']['outcome'] = event(current)
        result = summarize_cases([current], events=[], workflows=[saved])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_stale_assessment_excludes_amount_and_retains_history(self):
        old, new = case(), case(amount='0.00', digest='v2', status='NEEDS_REVIEW')
        result = summarize_cases([new], events=[event(old)],
                                 assessments=[assessment(old, validity='STALE')])
        self.assertEqual(result['counts']['reassessment'], 1)
        self.assertEqual(result['totals']['current_candidate'], '0.00')
        self.assertEqual(result['totals']['historical_stale_candidate'], '800.00')
        self.assertIsNone(result['cases'][0]['known_candidate'])

    def test_new_current_revision_replaces_800_with_1300_not_2100(self):
        old, new = case(), case(amount='1300.00', digest='v2')
        result = summarize_cases([old, new], events=[event(old, 'NEEDS_MORE_EVIDENCE')],
            assessments=[assessment(new, sequence=2), assessment(old, validity='STALE')])
        self.assertEqual(result['case_count'], 1)
        self.assertEqual(result['totals']['current_candidate'], '1300.00')
        self.assertEqual(result['totals']['historical_stale_candidate'], '0.00')
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['cases'][0]['assessment_id'], 'assessment-2')

    def test_current_assessment_with_mismatching_digest_requires_reassessment(self):
        old, new = case(), case(amount='1300.00', digest='v2')
        result = summarize_cases([new], events=[], assessments=[assessment(old)])
        self.assertEqual(result['counts']['reassessment'], 1)
        self.assertEqual(result['totals']['current_candidate'], '0.00')

    def test_explicit_reassessment_prepares_fresh_review_instead_of_repeating_reassessment(self):
        old, new = case(), case(amount='1300.00', digest='v2')
        result = summarize_cases([new], events=[event(old)], workflows=[workflow(old)],
                                 assessments=[assessment(old, validity='STALE'),
                                              assessment(new, sequence=2)])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['counts']['reassessment'], 0)
        self.assertEqual(result['totals']['current_candidate'], '1300.00')
        self.assertEqual(result['totals']['approved_draft'], '0.00')
        self.assertIsNone(result['cases'][0]['current_decision'])

    def test_same_digest_new_assessment_does_not_inherit_older_review(self):
        current = case()
        original = assessment(current)
        revised = {**assessment(current, sequence=2),
                   'prior_assessment_id': original['assessment_id'],
                   'created_at': '2026-10-17T13:00:00+00:00'}
        prior = workflow(current)
        prior['created_at'] = '2026-10-17T11:00:00+00:00'
        prior['state']['assessment_id'] = original['assessment_id']
        for events in ([], [event(current)]):
            with self.subTest(has_decision=bool(events)):
                result = summarize_cases([current], events=events, workflows=[prior],
                                         assessments=[original, revised])
                self.assertEqual(result['counts']['preparation'], 1)
                self.assertEqual(result['counts']['approved'], 0)
                self.assertEqual(result['counts']['approval'], 0)
                self.assertEqual(result['counts']['reassessment'], 0)
                self.assertEqual(result['totals']['current_candidate'], '800.00')
                self.assertEqual(result['totals']['approved_draft'], '0.00')
                self.assertIsNone(result['cases'][0]['current_decision'])

    def test_legacy_review_is_preserved_for_matching_migration_baseline(self):
        current = case()
        baseline = {**assessment(current), 'prior_assessment_id': None,
                    'created_at': '2026-10-17T14:00:00+00:00'}
        legacy = workflow(current)
        legacy['created_at'] = '2026-10-17T11:00:00+00:00'
        result = summarize_cases([current], events=[event(current)], workflows=[legacy],
                                 assessments=[baseline])
        self.assertEqual(result['counts']['approved'], 1)
        self.assertEqual(result['totals']['approved_draft'], '800.00')
        self.assertEqual(result['cases'][0]['current_decision'], 'APPROVED_FOR_DRAFT')

    def test_legacy_review_cannot_inherit_explicit_new_revision_with_same_digest(self):
        current = case()
        revised = {**assessment(current, sequence=2),
                   'prior_assessment_id': 'assessment-1',
                   'created_at': '2026-10-17T13:00:00+00:00'}
        legacy = workflow(current)
        legacy['created_at'] = '2026-10-17T11:00:00+00:00'
        result = summarize_cases([current], events=[event(current)], workflows=[legacy],
                                 assessments=[revised])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_fresh_matching_assessment_workflow_restores_approval_state(self):
        current = case()
        revised = {**assessment(current, sequence=2),
                   'prior_assessment_id': 'assessment-1',
                   'created_at': '2026-10-17T13:00:00+00:00'}
        old = workflow(current)
        old['state']['assessment_id'] = 'assessment-1'
        old['created_at'] = '2026-10-17T11:00:00+00:00'
        new = workflow(current, thread='new-linked-thread')
        new['state']['assessment_id'] = revised['assessment_id']
        new['created_at'] = '2026-10-17T14:00:00+00:00'
        prior_event = event(current)
        result = summarize_cases([current], events=[prior_event], workflows=[new, old],
                                 assessments=[revised])
        self.assertEqual(result['counts']['approval'], 1)
        self.assertEqual(result['totals']['approved_draft'], '0.00')
        latest_event = event(current, thread=new['thread_id'],
                             recorded_at='2026-10-17T15:00:00+00:00')
        result = summarize_cases([current], events=[latest_event, prior_event],
                                 workflows=[old, new], assessments=[revised])
        self.assertEqual(result['counts']['approved'], 1)
        self.assertEqual(result['counts']['approval'], 0)
        self.assertEqual(result['totals']['approved_draft'], '800.00')
        self.assertEqual(result['cases'][0]['workflow_thread_id'], new['thread_id'])

    def test_journal_without_workflow_cannot_alias_explicit_assessment_revision(self):
        current = case()
        revised = {**assessment(current, sequence=2), 'prior_assessment_id': 'assessment-1'}
        result = summarize_cases([current], events=[event(current)], assessments=[revised])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_needs_review_is_unknown_not_zero_saving(self):
        result = summarize_cases([case(amount='0.00', status='NEEDS_REVIEW')], events=[])
        self.assertEqual(result['counts']['evidence'], 1)
        self.assertIsNone(result['cases'][0]['known_candidate'])
        self.assertEqual(result['buckets']['evidence']['unknown_count'], 1)
        self.assertIsNone(result['totals']['expected_cash'])

    def test_invalid_candidate_amounts_do_not_turn_into_zero_or_float_math(self):
        for value in [None, 1.1, True, 'NaN', 'Infinity', '-1.00', 'invalid']:
            with self.subTest(value=value):
                result = summarize_cases([case(amount=value)], events=[])
                self.assertIsNone(result['cases'][0]['known_candidate'])
                self.assertEqual(result['buckets']['preparation']['unknown_count'], 1)
                self.assertEqual(result['totals']['current_candidate'], '0.00')

    def test_decimal_cents_remain_exact(self):
        result = summarize_cases([case('A', '0.10'), case('B', '0.20')], events=[])
        self.assertEqual(result['totals']['current_candidate'], '0.30')

    def test_currencies_have_separate_subtotals_and_usd_is_default(self):
        usd, eur = case(amount='800.00'), case('L002', '1300.00', currency='EUR')
        result = summarize_cases([usd, eur], events=[event(eur)])
        self.assertEqual(result['totals']['current_candidate'], '800.00')
        self.assertEqual(result['per_currency']['EUR']['current_candidate'], '1300.00')
        self.assertEqual(result['per_currency']['EUR']['approved_draft'], '1300.00')
        self.assertEqual(result['totals']['approved_draft'], '0.00')
        self.assertEqual(result['counts']['approved'], 1)

    def test_wrong_tenant_records_never_authorize_or_block_case(self):
        current, wrong = case(), case(tenant='OTHER')
        result = summarize_cases([current], events=[event(wrong)], workflows=[workflow(wrong)],
            requests=[{'tenant_id': 'OTHER', 'line_id': 'L001', 'status': 'OPEN'}],
            assessments=[assessment(wrong, validity='STALE')])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['request_counts']['open'], 0)
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_historical_currency_is_not_relabelled_after_scope_change(self):
        old, new = case(currency='EUR'), case(amount='1300.00', digest='v2')
        result = summarize_cases([new], events=[], assessments=[assessment(old, validity='STALE')])
        self.assertEqual(result['totals']['historical_stale_candidate'], '0.00')
        self.assertEqual(result['per_currency']['EUR']['historical_stale_candidate'], '800.00')
        self.assertEqual(result['cases'][0]['historical_currency'], 'EUR')

    def test_latest_journal_timestamp_respects_timezone_offsets(self):
        current = case()
        later = event(current, 'REJECTED', recorded_at='2026-10-17T10:00:00-07:00')
        earlier = event(current, recorded_at='2026-10-17T16:00:00+00:00')
        result = summarize_cases([current], events=[later, earlier])
        self.assertEqual(result['counts']['rejected'], 1)

    def test_open_and_attached_requests_block_without_fabricated_resolution(self):
        current = case()
        for status in ('OPEN', 'EVIDENCE_ATTACHED'):
            with self.subTest(status=status):
                request = {'request_id': 'R1', 'tenant_id': 'DEMO', 'line_id': 'L001',
                           'assessment_id': 'prior-assessment', 'status': status}
                result = summarize_cases([current], events=[event(current)], requests=[request])
                self.assertEqual(result['counts']['evidence'], 1)
                self.assertEqual(result['totals']['held_candidate'], '800.00')
                self.assertEqual(result['totals']['approved_draft'], '0.00')
                self.assertEqual(result['request_counts']['blocking_cases'], 1)

    def test_request_count_is_separate_from_case_count_and_deduplicated(self):
        current = case()
        requests = [dict(request_id='R1', tenant_id='DEMO', line_id='L001', status='OPEN'),
                    dict(request_id='R1', tenant_id='DEMO', line_id='L001', status='RESOLVED'),
                    dict(request_id='R2', tenant_id='DEMO', line_id='L001', status='OPEN'),
                    dict(request_id='R3', tenant_id='DEMO', line_id='L001', status='OPEN')]
        result = summarize_cases([current], events=[], requests=requests)
        self.assertEqual(result['case_count'], 1)
        self.assertEqual(result['request_counts'],
                         dict(open=2, evidence_attached=0, resolved=1, blocking_cases=1))

    def test_term_approval_cannot_become_invoice_case_approval(self):
        current = case()
        result = summarize_cases([current], events=[event(current, 'TERMS_APPROVED')])
        self.assertEqual(result['counts']['preparation'], 1)
        self.assertEqual(result['totals']['approved_draft'], '0.00')

    def test_inputs_are_not_modified(self):
        current = case()
        inputs = dict(cases=[current], events=[event(current)], workflows=[workflow(current)],
                      assessments=[assessment(current)], requests=[])
        before = deepcopy(inputs)
        summarize_cases(**inputs)
        self.assertEqual(inputs, before)

    def test_empty_dataset_still_reports_zero_realized_and_no_forecast(self):
        result = summarize_cases([], events=[])
        self.assertEqual(result['case_count'], 0)
        self.assertEqual(sum(result['counts'].values()), 0)
        self.assertEqual(result['totals']['current_candidate'], '0.00')
        self.assertEqual(result['totals']['realized_recovery'], '0.00')
        self.assertIsNone(result['totals']['expected_cash'])


if __name__ == '__main__':
    unittest.main()
