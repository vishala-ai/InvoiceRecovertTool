"""Create the user-authorized isolated 2026 portfolio and locked synthetic archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from recovery.demo_portfolio import create_scaled_portfolio


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'var/million-dollar-demo/million.sqlite3')
    parser.add_argument('--output', type=Path, default=ROOT / 'var/scaled-demo')
    args = parser.parse_args()
    def progress(event):
        print(json.dumps(event), file=sys.stderr, flush=True)
    result = create_scaled_portfolio(args.source, args.output, progress=progress)
    print(json.dumps({key: result[key] for key in ('status', 'current_invoice_lines', 'new_case_cohort',
        'current_counts', 'candidate_total_usd', 'approved_for_draft_usd', 'new_realized_recovery',
        'original_current_approvals_preserved', 'source_unchanged', 'model_calls', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
