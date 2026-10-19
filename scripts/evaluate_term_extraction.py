"""Opt-in local synthetic extraction smoke; never approves terms or cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from recovery.engine import audit
from recovery.lifecycle import generate_proposal, migrate
from recovery.storage import connect, load_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-local',action='store_true')
    parser.add_argument('--model',default='llama3.2:latest')
    parser.add_argument('--output',default='var/step9-implementation/live-term-smoke.json')
    args=parser.parse_args()
    result={'status':'NOT_RUN','provider':'ollama','model':args.model,
            'generation_attempts':0,'new_realized_recovery':'0.00','approvals_recorded':0}
    if args.run_local:
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'synthetic.sqlite3'
            load_snapshot(db,Path(__file__).resolve().parents[1]/'data')
            migrate(db,backup_dir=Path(folder)/'backups')
            with connect(db) as conn:
                before=audit(conn)
                terms=[tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms ORDER BY tenant_id,term_id')]
            result['generation_attempts']=1
            try:
                proposal=generate_proposal(db,'DEMO','DOC-A2',model=args.model)
                result.update(status='PASSED',proposal=proposal,
                    scope='local transport, strict schema, source quote membership and unreviewed persistence',
                    general_extraction_accuracy='NOT_ESTABLISHED')
            except (ValueError,RuntimeError,ImportError,OSError) as exc:
                result.update(status='WITHHELD',error=str(exc))
            with connect(db) as conn:
                after=audit(conn)
                assert before==after
                assert terms==[tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms ORDER BY tenant_id,term_id')]
                assert conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0]==0
                assert conn.execute('SELECT count(*) FROM review_events').fetchone()[0]==0
            result.update(financial_audit_unchanged=True,approved_terms_unchanged=True,
                          candidate_total=after['candidate_total'],case_count=after['case_count'])
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='proposal'},indent=2))
    return 0 if result['status'] in ('PASSED','NOT_RUN') else 1


if __name__=='__main__':
    raise SystemExit(main())
