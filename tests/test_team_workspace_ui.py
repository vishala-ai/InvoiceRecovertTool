"""Fictional team choices apply only to future real synthetic journal writes."""
from pathlib import Path
import json
import unittest

import test_ui as ui_helpers
from test_semantic_ui import database_contents

from recovery.lifecycle import migrate
from recovery.storage import connect
from recovery.team import ACCOUNTANTS, MANAGERS, journal_activity, reviewer_label
from recovery.workflow import run_persistent
from recovery.workspace_ui import DESCRIPTIONS, NAVIGATION, TEAM_PAGE


@unittest.skipUnless(ui_helpers.AVAILABLE, 'Install UI/graph extras for workspace and team checks')
class TeamWorkspaceUITests(unittest.TestCase):
    def setUp(self):
        self.helper = ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.db, self.cp = self.helper.db, self.helper.checkpoints
        migrate(self.db, backup_dir=Path(self.helper.temp.name) / 'backups')
        self.app = self.helper.app

    def clean(self):
        self.helper.assert_clean()
        self.assertFalse(self.app.error, [item.value for item in self.app.error])
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()

    def prior_approval(self):
        run_persistent(self.db, self.cp, 'existing:L001', line_id='L001', use_llm=False)
        run_persistent(self.db, self.cp, 'existing:L001', decision=dict(decision='approve',
            reviewer='Existing demonstration reviewer', note='Previously reviewed synthetic amendment.'))

    def test_button_navigation_and_page_descriptions_are_read_only(self):
        before = database_contents(self.db)
        self.app.run()
        for page, _, _ in NAVIGATION:
            with self.subTest(page=page):
                self.helper.navigate(page)
                self.clean()
                self.assertFalse(any(item.label == 'Navigation' for item in self.app.radio))
                navigation = [button for button in self.app.sidebar.button
                              if button.key and button.key.startswith('navigation:')]
                self.assertEqual(len(navigation), len(NAVIGATION))
                active = [button.key for button in navigation if button.proto.type == 'primary']
                self.assertEqual(active, ['navigation:' + page])
                self.assertIn(DESCRIPTIONS[page], [item.value for item in self.app.caption])
        self.assertEqual(database_contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.assertEqual(self.helper.events(), [])

    def test_team_has_two_managers_ten_accountants_and_no_invented_activity(self):
        self.app.run()
        self.helper.navigate(TEAM_PAGE)
        self.assertEqual([person['role'] for person in MANAGERS], ['Audit Manager', 'Finance Escalation Manager'])
        self.assertEqual(len(ACCOUNTANTS), 10)
        table = next(frame.value for frame in self.app.dataframe if 'Accountant ID' in frame.value.columns)
        self.assertEqual(table['Accountant ID'].tolist(), [f'ACC-{index:03d}' for index in range(1, 11)])
        self.assertTrue(table['Name'].str.contains('fictional').all())
        self.assertEqual({item.label: item.value for item in self.app.metric}['Recorded decisions'], '0')
        self.assertTrue(any('directory does not create approval activity' in item.value for item in self.app.info))
        self.assertTrue(any('not logins' in item.value for item in self.app.info))
        self.assertEqual(journal_activity(self.db)['events'], [])
        self.clean()

    def test_activity_keeps_real_previous_reviewer_and_original_notes(self):
        self.prior_approval()
        before = (database_contents(self.db), database_contents(self.cp))
        self.app.run()
        self.helper.navigate(TEAM_PAGE)
        table = next(frame.value for frame in self.app.dataframe if 'Recorded reviewer' in frame.value.columns)
        self.assertEqual(table['Recorded reviewer'].tolist(), ['Existing demonstration reviewer'])
        self.assertEqual(table['Attribution'].tolist(), ['Original demo reviewer label'])
        self.assertEqual(table['Review note'].tolist(), ['Previously reviewed synthetic amendment.'])
        self.helper.widget('selectbox', 'Activity reviewer').select('ACC-004').run()
        self.assertTrue(any('No saved decisions match' in item.value for item in self.app.info))
        self.assertEqual((database_contents(self.db), database_contents(self.cp)), before)
        self.clean()

    def test_directory_and_activity_define_every_column_without_changing_saved_events(self):
        self.prior_approval()
        before = (database_contents(self.db), database_contents(self.cp))
        self.app.run()
        self.helper.navigate(TEAM_PAGE)
        self.assertEqual(len(self.app.dataframe), 2)
        for table in self.app.dataframe:
            config = json.loads(table.proto.columns)
            self.assertTrue(config.pop('_index')['hidden'])
            self.assertEqual(set(config), set(table.value.columns))
            for column in table.value.columns:
                self.assertTrue(config[column]['help'])
                self.assertNotIn('Recorded field for the matching record', config[column]['help'])
        self.assertTrue(all('not a count of distinct current cases' in metric.proto.help
                            for metric in self.app.metric))
        self.assertTrue(any('an older approval does not establish a case' in item.value
                            for item in self.app.caption))
        self.assertEqual((database_contents(self.db), database_contents(self.cp)), before)
        self.clean()

    def test_roster_choice_records_selected_accountant_only_for_future_approval(self):
        self.prior_approval()
        run_persistent(self.db, self.cp, 'future:L004', line_id='L004', use_llm=False)
        before = journal_activity(self.db)['events']
        self.app.run()
        self.helper.navigate(TEAM_PAGE)
        self.helper.widget('selectbox', 'Accountant for future approvals').select('ACC-004').run()
        self.assertEqual(journal_activity(self.db)['events'], before)
        self.helper.navigate('Invoice review')
        self.helper.open_line('L004')
        self.assertEqual(self.helper.widget('selectbox', 'Approval accountant').value, 'ACC-004')
        reviewer = self.helper.widget('text_input', 'Reviewer label')
        self.assertTrue(reviewer.disabled)
        self.assertEqual(reviewer.value, reviewer_label('ACC-004'))
        self.helper.widget('text_area', 'Evidence review note').set_value('Checked the selected fictional accountant and complete evidence.')
        self.helper.widget('button', 'Approve for draft').click().run()
        self.clean()
        events = self.helper.events()
        self.assertEqual(len(events), 2)
        self.assertEqual(next(event for event in events if event['line_id'] == 'L001')['reviewer'],
                         'Existing demonstration reviewer')
        self.assertEqual(next(event for event in events if event['line_id'] == 'L004')['reviewer'],
                         reviewer_label('ACC-004'))
        self.helper.navigate(TEAM_PAGE)
        self.helper.widget('selectbox', 'Activity reviewer').select('ACC-004').run()
        table = next(frame.value for frame in self.app.dataframe if 'Recorded reviewer' in frame.value.columns)
        self.assertEqual(table['Invoice line'].tolist(), ['L004'])
        self.assertEqual(table['Attribution'].tolist(), ['Fictional directory label'])
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()
