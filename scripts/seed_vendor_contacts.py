"""Explicitly add fictional vendor POC names to an existing synthetic database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from recovery.vendor_contacts import seed_vendor_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--tenant', default='DEMO')
    parser.add_argument('--synthetic-confirmed', action='store_true', required=True,
                        help='Confirm that every generated POC name is fictional demo data.')
    args = parser.parse_args()
    try:
        report = seed_vendor_contacts(args.db, tenant=args.tenant,
                                      synthetic_confirmed=args.synthetic_confirmed)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({'status': 'REJECTED', 'message': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
