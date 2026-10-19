"""Resolve chat links against saved scoped records before selecting a UI view."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import uuid

from .storage import metadata


def resolve_chat_target(db, target, *, tenant='DEMO', kind='contract'):
    """A chat response is never authority to select another tenant's record."""
    if (not isinstance(target, dict) or kind not in {'contract', 'case'}
            or not isinstance(tenant, str) or not tenant.strip()):
        raise ValueError('The chat link is invalid; search for the record again.')
    for field in ('supplier_id', 'contract_id', 'line_id'):
        value = target.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()
                                  or len(value) > 200):
            raise ValueError('The chat link is invalid; search for the record again.')
    if kind == 'case' and not target.get('line_id'):
        raise ValueError('A case link requires an invoice line.')
    if kind == 'contract' and not target.get('supplier_id'):
        raise ValueError('A vendor or contract link requires a vendor.')
    year = target.get('origin_year')
    if year is not None and (type(year) is not int or year not in (2024, 2025)):
        raise ValueError('The historical year is unavailable.')
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        manifest = json.loads(metadata(conn, 'manifest'))
        if not isinstance(manifest, dict) or manifest.get('synthetic') is not True:
            raise ValueError('Only the synthetic workspace is supported.')
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        field = 'line_id' if kind == 'case' else 'supplier_id'
        value = target[field]
        where, args = f'tenant_id=? AND {field}=?', [tenant, value]
        if kind == 'contract' and target.get('contract_id'):
            where += ' AND contract_id=?'
            args.append(target['contract_id'])
        row = None
        if year is None:
            row = conn.execute('SELECT supplier_id,contract_id,line_id FROM invoice_lines WHERE '
                               + where + ' ORDER BY line_id LIMIT 1', args).fetchone()
            if row is None and kind == 'contract':
                row = conn.execute('SELECT supplier_id,contract_id,NULL AS line_id FROM documents WHERE '
                                   + where + ' ORDER BY doc_id LIMIT 1', args).fetchone()
        historical = row is None and 'legacy_cases' in tables
        if historical:
            if year is not None:
                where += ' AND origin_year=?'
                args.append(year)
            row = conn.execute('SELECT supplier_id,contract_id,line_id,origin_year FROM legacy_cases WHERE '
                               + where + ' ORDER BY origin_year DESC,line_id LIMIT 1', args).fetchone()
        if row is None:
            raise ValueError('That scoped record is unavailable. Search again.')
        result = dict(row)
        if any(target.get(key) is not None and target[key] != result[key]
               for key in ('supplier_id', 'contract_id')):
            raise ValueError('The chat link scope does not match the saved record.')
        result['origin_year'] = result.get('origin_year') if historical else None
        if kind == 'contract':
            result['line_id'] = None
            # A vendor link should show all of its contracts and cases.
            result['contract_id'] = target.get('contract_id')
        result['token'] = uuid.uuid4().hex
        return result


def consume_chat_target(st, db, *, tenant='DEMO', kind='contract', namespace='contracts'):
    resolved = str(Path(db).resolve())
    key = f'chatbot_open_{kind}:{resolved}:{tenant}'
    target = st.session_state.pop(key, None)
    active_key = f'chatbot_active_target:{resolved}:{tenant}:{namespace}'
    if target is not None:
        try:
            active = resolve_chat_target(db, target, tenant=tenant, kind=kind)
        except (ValueError, OSError, sqlite3.Error):
            st.warning('That scoped record is unavailable. Search again in the assistant.')
            st.session_state.pop(active_key, None)
        else:
            st.session_state[active_key] = active
            if namespace == 'contracts':
                st.session_state.pop('lifecycle:terms:document', None)
            year = active['origin_year']
            st.session_state[f'{namespace}_cohort:{resolved}'] = (
                f'Migrated · {year} (locked)' if year else 'Current · 2026')
    return st.session_state.get(active_key)
