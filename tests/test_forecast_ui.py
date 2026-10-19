"""Read-only Overview uses saved assumptions without exposing the assumption editor."""
from pathlib import Path
import unittest

import test_ui as ui_helpers
from test_semantic_ui import database_contents
from test_scaled_portfolio_ui import archive_payload

from recovery.forecast import LABEL, get_forecast, save_forecast
from recovery.legacy_archive import import_archive
from recovery.lifecycle import migrate, reassess
from recovery.storage import connect
from recovery.workflow import run_persistent


@unittest.skipUnless(ui_helpers.AVAILABLE, 'Install UI/graph extras for forecast UI checks')
class ForecastUITests(unittest.TestCase):
    def setUp(self):
        self.helper = ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.db, self.cp = self.helper.db, self.helper.checkpoints
        migrate(self.db, backup_dir=Path(self.helper.temp.name) / 'backups')
        run_persistent(self.db, self.cp, 'forecast:L004', line_id='L004', use_llm=False)
        run_persistent(self.db, self.cp, 'forecast:L004', decision=dict(decision='approve',
            reviewer='Existing forecast-basis reviewer', note='Checked the synthetic USD 600 case.'))
        self.app = self.helper.app

    def overview(self):
        self.app.run()
        self.helper.navigate('Overview')
        self.helper.assert_clean()
        self.assertFalse(self.app.error)

    def metric(self):
        return next(item.value for item in self.app.metric if item.label == LABEL)

    def save(self, percentage='80'):
        # Preserve integration checks of the reusable editor in a separate test
        # harness. The accountant's Overview must not expose this editor.
        from streamlit.testing.v1 import AppTest
        self.editor = AppTest.from_string('''
import os
from pathlib import Path
import streamlit as st
from recovery.forecast import get_forecast
from recovery.forecast_ui import render_forecast_assumption
from recovery.historical_recovery import get_recovery_overview
db = Path(os.environ['RECOVERY_DB'])
cp = db.with_name(db.stem + '-ui-checkpoints.sqlite3')
forecast = get_forecast(db, cp, scope='all_approved')
render_forecast_assumption(st, db, cp, forecast, scope='all_approved',
    recovery_overview=get_recovery_overview(db, cp))
''', default_timeout=30).run()
        def widget(kind, label):
            return next(item for item in getattr(self.editor, kind) if item.label == label)
        for label, value in (
            ('Assumed recovery percentage (0–100)', percentage),
            ('Forecast reviewer label', 'Synthetic forecast reviewer'),
            ('Forecast as-of date (YYYY-MM-DD)', '2026-10-18'),
            ('Forecast horizon (YYYY-MM-DD)', '2026-12-31')):
            widget('text_input', label).set_value(value)
        widget('text_area', 'Reason for this demo assumption').set_value(
            'Illustrative uncalibrated cash-collection assumption for this synthetic example.')
        widget('button', 'Save reviewed forecast assumption').click().run()
        self.assertFalse(self.editor.exception)
        self.app.run()
        self.helper.assert_clean()

    def test_unset_forecast_render_is_read_only_and_does_not_save_default(self):
        before = (database_contents(self.db), database_contents(self.cp))
        self.overview()
        self.assertEqual(self.metric(), 'Not estimated')
        metrics = {item.label: item.value for item in self.app.metric}
        self.assertEqual(metrics['Total approved records'], '1')
        self.assertEqual(metrics['Current approved for draft'], '1')
        self.assertTrue(any(item.value == '1 current / 0 historical' for item in self.app.caption))
        self.assertFalse(self.app.main.text_input)
        self.assertFalse(self.app.text_area)
        self.assertFalse(self.app.expander)
        self.assertFalse(any(button.label == 'Save reviewed forecast assumption'
                             for button in self.app.button))
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved')['status'], 'NOT_SET')
        self.assertEqual((database_contents(self.db), database_contents(self.cp)), before)
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()

    def test_save_shows_reviewed_decimal_projection_and_preserves_recovery_facts(self):
        events = self.helper.events()
        self.overview()
        self.save()
        self.assertFalse(self.app.error)
        self.assertEqual(self.metric(), 'USD 480.00')
        saved = get_forecast(self.db, self.cp, scope='all_approved')
        self.assertEqual(saved['basis']['amount_usd'], '600.00')
        self.assertEqual(saved['basis']['case_count'], 1)
        self.assertEqual(saved['assumption']['recovery_percentage'], '80')
        self.assertEqual(saved['assumption']['reviewer'], 'Synthetic forecast reviewer')
        self.assertEqual(self.helper.events(), events)
        self.app.run()
        self.assertEqual(self.metric(), 'USD 480.00')
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()

    def test_changed_approved_basis_withholds_projection_until_reviewed_again(self):
        self.overview()
        self.save()
        reassess(self.db, 'DEMO', 'L004', idempotency_key='forecast-new-assessment')
        self.app.run()
        self.assertEqual(self.metric(), 'Not estimated')
        self.assertTrue(any('approved-case basis changed' in item.value for item in self.app.warning))
        saved = get_forecast(self.db, self.cp, scope='all_approved')
        self.assertEqual(saved['status'], 'STALE_BASIS')
        self.assertEqual(saved['assumption']['projected_cash_usd'], '480.00')
        self.assertIsNone(saved['projected_cash_usd'])
        self.assertFalse(any(button.label == 'Save reviewed forecast assumption'
                             for button in self.app.button))

    def test_invalid_percentage_is_rejected_without_a_forecast_or_review_event(self):
        events = self.helper.events()
        self.overview()
        self.save('80%')
        self.assertTrue(self.editor.error)
        self.assertFalse(self.app.error)
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved')['status'], 'NOT_SET')
        self.assertEqual(self.helper.events(), events)
        self.assertEqual(self.metric(), 'Not estimated')

    def test_explicit_zero_percentage_remains_a_current_estimate(self):
        self.overview()
        self.save('0')
        self.assertFalse(self.app.error)
        self.assertEqual(self.metric(), 'USD 0.00')
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved')['status'], 'CURRENT')

    def test_returning_to_an_earlier_percentage_creates_a_new_reviewed_assumption(self):
        self.overview()
        for percentage, expected in (('80', 'USD 480.00'), ('50', 'USD 300.00'), ('80', 'USD 480.00')):
            self.save(percentage)
            self.assertFalse(self.app.error)
            self.assertEqual(self.metric(), expected)
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved')['assumption']['recovery_percentage'], '80')
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM forecast_all_approved_assumptions').fetchone()[0], 3)

    def test_overview_restores_coverage_and_workload_beside_unchanged_grand_totals(self):
        import_archive(self.db, archive_payload())
        before = (database_contents(self.db), database_contents(self.cp))
        self.overview()
        metrics = {item.label: item.value for item in self.app.metric}
        self.assertEqual({label: metrics[label] for label in ('Total approved value', LABEL, 'Realized recovery')},
                         {'Total approved value': 'USD 2,167.18',
                          LABEL: 'Not estimated', 'Realized recovery': 'USD 0.00'})
        self.assertEqual((metrics['Suppliers'], metrics['Contracts']), ('5', '6'))
        self.assertEqual(metrics['Total approved records'], '128')
        self.assertEqual(metrics['Current approved for draft'], '1')
        self.assertTrue(any(item.value == '1 current / 127 historical' for item in self.app.caption))
        self.assertFalse(self.app.dataframe)
        self.assertEqual(sum(button.label == 'View cases' for button in self.app.button), 6)
        self.assertFalse(self.app.expander)
        self.assertFalse(self.app.main.text_input)
        self.assertEqual([item.value for item in self.app.subheader],
                         ['GRAND TOTALS (USD)', 'PORTFOLIO COVERAGE', 'REVIEW WORKLOAD'])
        self.assertTrue(any('separate measures across all years' in item.value
                            for item in self.app.caption))
        self.assertFalse(any('Locked migration archive' in item.value for item in self.app.info))
        self.assertEqual((database_contents(self.db), database_contents(self.cp)), before)
        self.save()
        self.assertEqual(self.metric(), 'USD 1,733.74')
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved')['basis']['case_count'], 128)
        self.assertEqual(len(self.helper.events()), 1)

    def test_current_only_assumption_does_not_automatically_apply_to_grand_totals(self):
        current = get_forecast(self.db, self.cp)
        save_forecast(self.db, self.cp, expected_basis_digest=current['basis']['basis_digest'],
            recovery_percentage='80', reason='Prior reviewed current-only demo assumption.',
            reviewer='Prior forecast reviewer', as_of='2026-10-18', horizon_date='2026-12-31',
            idempotency_key='prior-current-assumption')
        self.overview()
        self.assertEqual(self.metric(), 'Not estimated')
        self.assertEqual(get_forecast(self.db, self.cp)['status'], 'CURRENT')
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved')['status'], 'NOT_SET')
        self.save()
        self.assertEqual(self.metric(), 'USD 480.00')
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM forecast_assumptions').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM forecast_all_approved_assumptions').fetchone()[0], 1)

    def test_historical_demo_receipts_move_to_realized_without_double_count_or_new_reviews(self):
        from recovery.historical_recovery import save_historical_realization

        import_archive(self.db, archive_payload())
        self.overview()
        self.save()
        frozen = get_forecast(self.db, self.cp, scope='all_approved')
        events = self.helper.events()
        save_historical_realization(self.db, self.cp,
            expected_basis_digest=frozen['basis']['basis_digest'],
            reviewer='User-authorized historical demo fixture owner',
            reason='Invented 2024 and 2025 receipts at 100% of the original 80% forecast.',
            as_of='2026-10-18', idempotency_key='ui-historical-demo', synthetic_confirmed=True)
        before = (database_contents(self.db), database_contents(self.cp))
        self.app.run()
        self.helper.assert_clean()
        self.assertFalse(self.app.error)
        metrics = {item.label: item.value for item in self.app.metric}
        self.assertEqual({label: metrics[label] for label in ('Total approved value', LABEL, 'Realized recovery')}, {
            'Total approved value': 'USD 2,167.18', LABEL: 'USD 480.00',
            'Realized recovery': 'USD 1,253.74'})
        self.assertEqual((metrics['Suppliers'], metrics['Contracts']), ('5', '6'))
        self.assertFalse(self.app.expander)
        self.assertTrue(any('invented 2024–2025 demo receipts' in item.value for item in self.app.caption))
        self.assertTrue(any('80% × USD 600.00' in item.value for item in self.app.caption))
        self.assertEqual((database_contents(self.db), database_contents(self.cp)), before)
        self.assertEqual(get_forecast(self.db, self.cp, scope='all_approved'), frozen)
        self.save('90')
        self.assertEqual(self.metric(), 'USD 540.00')
        self.assertEqual(next(item.value for item in self.app.metric if item.label == 'Realized recovery'),
                         'USD 1,253.74')
        self.assertEqual(self.helper.events(), events)
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
