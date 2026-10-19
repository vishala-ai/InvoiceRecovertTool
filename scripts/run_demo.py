"""Launch a verified synthetic portfolio at one fixed local address.

Startup verifies the selected profile; it never imports data, builds an index,
calls a provider or creates approvals. Run with --check for read-only validation.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / 'var/million-dollar-demo/million.sqlite3'
INDEX = ROOT / 'var/million-dollar-demo/vector-index'
SCALED_DIRECTORY = ROOT / 'var/scaled-demo'
URL = 'http://127.0.0.1:8515/'


def _required_file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'The existing synthetic profile file is required: {path}. '
                         'Startup never creates or replaces profile files. See README.md.')


def verify_scaled_profile() -> dict:
    from recovery.demo_portfolio import AS_OF, PROFILE_ID
    from recovery.legacy_archive import archive_summary
    from recovery.storage import metadata

    database = SCALED_DIRECTORY / 'portfolio.sqlite3'
    checkpoints = SCALED_DIRECTORY / 'portfolio-ui-checkpoints.sqlite3'
    marker = SCALED_DIRECTORY / 'portfolio-profile.json'
    for path in (database, checkpoints, marker):
        _required_file(path)
    profile = json.loads(marker.read_text())
    if (profile.get('synthetic') is not True or profile.get('profile_id') != PROFILE_ID
            or profile.get('as_of') != AS_OF
            or profile.get('current_cohort') != {
                'cases': 1000, 'vendors': 35, 'approved': 500, 'awaiting': 250, 'evidence': 250}
            or profile.get('reconciliation', {}).get('matched') is not True):
        raise ValueError('The expanded synthetic profile marker is incomplete or unsupported')
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        manifest = json.loads(metadata(conn, 'manifest'))
        if manifest.get('synthetic') is not True or manifest.get('as_of') != AS_OF:
            raise ValueError('The selected database must attest the matching synthetic scope')
        if json.loads(metadata(conn, 'demo_portfolio')) != profile:
            raise ValueError('Expanded profile marker does not match the selected database')
        count = conn.execute("SELECT COUNT(*) FROM invoice_lines WHERE tenant_id='DEMO' "
                             "AND line_id LIKE 'SCALE-2026-%'").fetchone()[0]
        vendors = conn.execute("SELECT COUNT(*) FROM suppliers WHERE tenant_id='DEMO' "
                               "AND supplier_id LIKE 'DEMO-SCALE-V%'").fetchone()[0]
        archive = archive_summary(database, 'DEMO')
        if count != 1000 or vendors != 35 or archive['case_count'] != 2000:
            raise ValueError('The expanded synthetic portfolio is incomplete')
    return {'url': URL, 'database': str(database), 'checkpoints': str(checkpoints),
            'index': None, 'retrieval': 'SQLite FTS5; no semantic index selected',
            'synthetic_only': True, 'tenant': 'DEMO', 'dataset_id': PROFILE_ID,
            'current_added_cases': count, 'current_added_vendors': vendors,
            'locked_historical_cases': archive['case_count'],
            'validation': 'matching published synthetic profile; no provider call'}


def verify_profile(profile='scaled') -> dict:
    if profile == 'scaled':
        return verify_scaled_profile()
    if profile != 'million':
        raise ValueError('Unknown synthetic demo profile')
    if DATABASE.is_symlink() or not DATABASE.is_file():
        raise ValueError('The existing million-dollar synthetic database is required at '
                         f'{DATABASE}. Prepare or restore that separate demo first; '
                         'this launcher never initializes or replaces databases. See README.md.')
    if not INDEX.is_dir() or not (INDEX / 'evidence-index.json').is_file():
        raise ValueError('The dedicated million-dollar evidence index is required at '
                         f'{INDEX}. Build it explicitly with recover --db {DATABASE} '
                         f'index-evidence --index {INDEX} --embedding-model '
                         'YOUR_INSTALLED_LOCAL_EMBEDDING_MODEL --allow-local-embedding. '
                         'This launcher never builds indexes.')
    from recovery.semantic_retrieval import resolve_index_config, validate_index_config
    from recovery.storage import metadata

    # Read-only URI mode cannot create a missing/fallback database. Keep all
    # canonical-source checks in a single read snapshot, without provider calls.
    with closing(sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        manifest = json.loads(metadata(conn, 'manifest'))
        if not isinstance(manifest, dict) or manifest.get('synthetic') is not True:
            raise ValueError('The selected demo database must attest synthetic=true')
        config = resolve_index_config(INDEX)
        _, catalog, _, _ = validate_index_config(conn, 'DEMO', config)
    return {'url': URL, 'database': str(DATABASE), 'index': str(INDEX),
            'synthetic_only': True, 'tenant': 'DEMO', 'dataset_id': manifest['dataset_id'],
            'generation': config['generation'], 'embedding_model': catalog['embedding']['model'],
            'embedding_model_digest': catalog['embedding']['model_digest'],
            'validation': 'canonical source/index match; no provider call'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('scaled', 'million'), default='scaled',
                        help='Published synthetic portfolio (default: scaled)')
    parser.add_argument('--check', action='store_true',
                        help='Validate the fixed demo profile without starting Streamlit or calling a model')
    args = parser.parse_args(argv)
    try:
        profile = verify_profile(args.profile)
        if args.check:
            print(json.dumps(profile, indent=2))
            return 0
        environment = os.environ.copy()
        environment['RECOVERY_DB'] = profile['database']
        if profile['index'] is None:
            environment.pop('RECOVERY_RETRIEVAL_INDEX', None)
        else:
            environment['RECOVERY_RETRIEVAL_INDEX'] = profile['index']
        os.chdir(ROOT)
        print(f'Starting the existing synthetic demo at {URL}', flush=True)
        os.execvpe('uv', ['uv', 'run', '--all-extras', '--frozen', 'streamlit', 'run',
                         str(ROOT / 'app.py'), '--server.address', '127.0.0.1',
                         '--server.port', '8515', '--server.headless', 'true',
                         '--browser.gatherUsageStats', 'false'], environment)
    except (ValueError, RuntimeError, ImportError, OSError, sqlite3.Error) as exc:
        print(f'Demo launch stopped: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
