"""Server-side intake schemas fail cleanly before changing accepted evidence."""
from pathlib import Path
import tempfile
import unittest

from recovery import lifecycle
from recovery.storage import connect, load_snapshot


class LifecycleInputTests(unittest.TestCase):
    def test_missing_extra_and_malformed_scope_fields_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'synthetic.sqlite3'
            load_snapshot(db,Path(__file__).resolve().parents[1]/'data')
            lifecycle.migrate(db,backup_dir=Path(folder)/'backups')
            stage=lifecycle.stage_document(db,'DEMO','synthetic.txt',b'Synthetic rate evidence.',True)
            scope={'supplier_id':'SUP-A','contract_id':'CTR-A','role':'ENGINEER','location':'REMOTE'}
            invalid=[None,[],{}, {key:value for key,value in scope.items() if key!='role'},
                     {**scope,'currency':'USD'},{**scope,'role':True}]
            for index,value in enumerate(invalid):
                with self.subTest(scope=value),self.assertRaises(ValueError):
                    lifecycle.admit_document(db,'DEMO',stage['stage_id'],scope=value,
                        title='Synthetic source',signed_date='2026-07-20',status='SIGNED',
                        synthetic_confirmed=True,signed_simulation_confirmed=True,
                        reviewer='Demo reviewer',note='Malformed scope must be rejected.',
                        idempotency_key=f'bad-scope:{index}')
            with connect(db) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM source_versions WHERE origin!='CURATED_FIXTURE'").fetchone()[0],0)
                self.assertEqual(conn.execute('SELECT count(*) FROM operation_keys').fetchone()[0],0)


if __name__=='__main__':
    unittest.main()
