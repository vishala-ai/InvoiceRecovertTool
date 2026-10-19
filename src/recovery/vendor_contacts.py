"""Explicit fictional vendor contacts, separate from canonical financial inputs.

Reading never creates or updates a table. Only the explicit seed operation may
append contacts, after checking the synthetic manifest and exact vendor scope.
Contacts do not authorize supplier communications or enter case/forecast hashes.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


NOTICE = 'Fictional vendor points of contact for the local synthetic demonstration.'
PROVENANCE = 'fictional-local-demo-contact-v1'
_TABLE = 'vendor_contacts'
_FIELDS = ('tenant_id', 'supplier_id', 'poc_name', 'synthetic', 'provenance', 'created_at')
_FIRST = ('Avery', 'Jordan', 'Morgan', 'Taylor', 'Casey', 'Riley', 'Cameron', 'Quinn',
          'Parker', 'Jamie', 'Alex', 'Robin', 'Dana', 'Sam', 'Jesse', 'Reese')
_LAST = ('Bennett', 'Brooks', 'Carter', 'Collins', 'Davis', 'Ellis', 'Foster', 'Hayes',
         'Lane', 'Morgan', 'Parker', 'Reed', 'Shaw', 'Taylor', 'Walker', 'Wells')


def _text(value, field, maximum=200):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ValueError(f'Vendor contact {field} must be nonempty text without control characters.')
    return value.strip()


def _exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


@contextmanager
def _connection(db_or_connection, *, write=False):
    if isinstance(db_or_connection, sqlite3.Connection):
        yield db_or_connection
        return
    path = Path(db_or_connection)
    if not path.is_file():
        raise ValueError('An existing synthetic database is required for vendor contacts.')
    mode = 'rw' if write else 'ro'
    with closing(sqlite3.connect(path.resolve().as_uri() + f'?mode={mode}', uri=True)) as conn:
        yield conn


def _guard(conn, tenant):
    tenant = _text(tenant, 'tenant')
    if not _exists(conn, 'metadata'):
        raise ValueError('SYNTHETIC_ONLY: vendor contacts require a synthetic import manifest.')
    row = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    try:
        manifest = json.loads(row[0]) if row else None
    except (ValueError, TypeError) as exc:
        raise ValueError('SYNTHETIC_ONLY: vendor contact manifest is malformed.') from exc
    if not isinstance(manifest, dict) or manifest.get('synthetic') is not True:
        raise ValueError('SYNTHETIC_ONLY: vendor contacts require a synthetic demo database.')
    tenants = manifest.get('tenants')
    if not isinstance(tenants, dict) or tenant not in tenants:
        raise ValueError('Vendor contact tenant is not in the synthetic manifest.')
    return tenant


def _suppliers(conn, tenant):
    suppliers = {row[0] for row in conn.execute(
        'SELECT supplier_id FROM suppliers WHERE tenant_id=?', (tenant,))}
    if _exists(conn, 'legacy_suppliers'):
        rows = conn.execute('''SELECT s.supplier_id,b.synthetic FROM legacy_suppliers s
            LEFT JOIN legacy_archive_batches b ON s.archive_id=b.archive_id
            WHERE s.tenant_id=?''', (tenant,))
        for supplier_id, synthetic in rows:
            if synthetic != 1:
                raise ValueError('SYNTHETIC_ONLY: legacy vendor contact scope is unverified.')
            suppliers.add(supplier_id)
    return suppliers


def _saved(conn, tenant, suppliers):
    if not _exists(conn, _TABLE):
        return []
    columns = [row[1] for row in conn.execute('PRAGMA table_info(vendor_contacts)')]
    if tuple(columns) != _FIELDS:
        raise ValueError('Vendor contact schema is incompatible with the explicit seed operation.')
    rows = conn.execute('''SELECT tenant_id,supplier_id,poc_name,synthetic,provenance,created_at
        FROM vendor_contacts WHERE tenant_id=? ORDER BY supplier_id''', (tenant,))
    contacts = []
    for values in rows:
        row = dict(zip(_FIELDS, values))
        if (row['supplier_id'] not in suppliers or row['synthetic'] != 1
                or row['provenance'] != PROVENANCE):
            raise ValueError('Saved vendor contact has unknown scope or synthetic provenance.')
        if _text(row['poc_name'], 'name') != row['poc_name']:
            raise ValueError('Saved vendor contact name is not canonical.')
        try:
            created = datetime.fromisoformat(row['created_at'])
        except (ValueError, TypeError) as exc:
            raise ValueError('Saved vendor contact timestamp is invalid.') from exc
        if created.tzinfo is None:
            raise ValueError('Saved vendor contact timestamp requires a timezone.')
        contacts.append({**row, 'synthetic': True})
    return contacts


def list_vendor_contacts(db_or_connection, tenant='DEMO'):
    """Return saved scoped contacts; an absent contact table is an empty result.

    Database paths are opened read-only. A supplied SQLite connection retains its
    existing transaction and row factory. No provider or workflow is called.
    """
    with _connection(db_or_connection) as conn:
        tenant = _guard(conn, tenant)
        return _saved(conn, tenant, _suppliers(conn, tenant))


def get_vendor_contact(db_or_connection, tenant='DEMO', supplier_id=None):
    """Return a saved contact, or None for an unknown/unseeded scoped supplier."""
    supplier_id = _text(supplier_id, 'supplier ID')
    contacts = list_vendor_contacts(db_or_connection, tenant)
    return next((row for row in contacts if row['supplier_id'] == supplier_id), None)


def _fictional_name(tenant, supplier_id):
    digest = hashlib.sha256(f'{PROVENANCE}:{tenant}:{supplier_id}'.encode()).digest()
    return f'{_FIRST[digest[0] % len(_FIRST)]} {_LAST[digest[1] % len(_LAST)]}'


def _requested(contacts, tenant, suppliers):
    if contacts is None:
        return [{'supplier_id': supplier, 'poc_name': _fictional_name(tenant, supplier)}
                for supplier in sorted(suppliers)]
    if not isinstance(contacts, list):
        raise ValueError('Vendor contacts must be a list of supplier_id/poc_name objects.')
    result, seen = [], set()
    for row in contacts:
        if not isinstance(row, dict) or set(row) != {'supplier_id', 'poc_name'}:
            raise ValueError('Vendor contact input requires exactly supplier_id and poc_name.')
        supplier = _text(row['supplier_id'], 'supplier ID')
        name = _text(row['poc_name'], 'name')
        if supplier not in suppliers:
            raise ValueError('Vendor contact supplier is not in the selected tenant scope.')
        if supplier in seen:
            raise ValueError('Duplicate vendor contact supplier in seed input.')
        seen.add(supplier)
        result.append({'supplier_id': supplier, 'poc_name': name})
    return result


def seed_vendor_contacts(db_or_connection, *, tenant='DEMO', synthetic_confirmed=False,
                         contacts=None):
    """Atomically seed fictional names without replacing any existing contact.

    Omit contacts to cover every current and legacy supplier with deterministic
    invented names, preserving prior valid names. An explicit list with a changed
    saved name is rejected. An idle connection is required so this function never
    commits another operation's transaction. It modifies only vendor_contacts.
    """
    if synthetic_confirmed is not True:
        raise ValueError('SYNTHETIC_ONLY: explicitly confirm all contact names are fictional.')
    with _connection(db_or_connection, write=True) as conn:
        if conn.in_transaction:
            raise ValueError('Vendor contact seeding requires an idle database connection.')
        try:
            conn.execute('BEGIN IMMEDIATE')
            tenant = _guard(conn, tenant)
            suppliers = _suppliers(conn, tenant)
            requested = _requested(contacts, tenant, suppliers)
            prior = {row['supplier_id']: row for row in _saved(conn, tenant, suppliers)}
            for row in requested:
                old = prior.get(row['supplier_id'])
                if old and contacts is not None and old['poc_name'] != row['poc_name']:
                    raise ValueError('CONTACT_CONFLICT: existing fictional contact cannot be replaced.')
            conn.execute('''CREATE TABLE IF NOT EXISTS vendor_contacts (
                tenant_id TEXT NOT NULL,supplier_id TEXT NOT NULL,poc_name TEXT NOT NULL,
                synthetic INTEGER NOT NULL CHECK(synthetic=1),provenance TEXT NOT NULL,
                created_at TEXT NOT NULL,PRIMARY KEY(tenant_id,supplier_id))''')
            created_at = datetime.now(timezone.utc).isoformat()
            inserted = 0
            for row in requested:
                if row['supplier_id'] not in prior:
                    conn.execute('''INSERT INTO vendor_contacts
                        (tenant_id,supplier_id,poc_name,synthetic,provenance,created_at)
                        VALUES (?,?,?,1,?,?)''',
                        (tenant, row['supplier_id'], row['poc_name'], PROVENANCE, created_at))
                    inserted += 1
            saved = _saved(conn, tenant, suppliers)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return {'status': 'SEEDED', 'tenant_id': tenant, 'synthetic': True, 'notice': NOTICE,
            'supplier_count': len(suppliers), 'inserted': inserted,
            'retained': len(requested) - inserted, 'contacts': saved}
