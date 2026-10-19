"""Strict, standard-library schemas for the synthetic fixture format.

Validation never coerces or rewrites source values. Decimal fields accept at most
18 coefficient digits, six fractional places, and magnitudes below 10**12. Their
stored exponent must be between -6 and 12. These bounds also reject extreme zero
exponents before values reach the financial engine.
Business evidence conflicts (for example partial payments) belong to the engine,
not these format checks.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class Field:
    """A required CSV column, with optional blank values when ``optional`` is true."""

    kind: str
    optional: bool = False
    choices: tuple[str, ...] = ()


ID = Field('identifier')
TEXT = Field('text')
DATE = Field('date')
DECIMAL = Field('decimal')
CURRENCY = Field('currency')
BOOL = Field('bool')
UNIT = Field('enum', choices=('HOUR', 'DAY'))
SERVICE_TYPE = Field('enum', choices=('REGULAR', 'OVERTIME'))

# Dict insertion order is the canonical table/column order, independent of the
# source CSV's order. SQL names must come from this schema, never an input header.
TABLE_SCHEMAS: dict[str, dict[str, Field]] = {
    'suppliers': {
        'tenant_id': ID, 'supplier_id': ID, 'name': TEXT,
    },
    'documents': {
        'doc_id': ID, 'tenant_id': ID, 'supplier_id': ID, 'contract_id': ID,
        'role': ID, 'location': ID,
        'status': Field('enum', choices=('SIGNED', 'DRAFT', 'UNTRUSTED')),
        'kind': Field('enum', choices=('RATE_AGREEMENT', 'NOTE')),
        'signed_date': Field('date', optional=True),
        'path': Field('path'), 'sha256': Field('sha256'),
    },
    'reviewed_terms': {
        'term_id': ID, 'tenant_id': ID, 'supplier_id': ID, 'contract_id': ID,
        'role': ID, 'location': ID, 'currency': CURRENCY, 'unit': UNIT,
        'service_type': SERVICE_TYPE, 'effective_from': DATE,
        'effective_to': Field('date', optional=True), 'rate': DECIMAL,
        'supersedes': Field('identifier', optional=True), 'doc_id': ID,
        'source_quote': TEXT, 'reviewer': Field('text', optional=True),
        'reviewed': BOOL,
    },
    'invoice_lines': {
        'tenant_id': ID, 'line_id': ID, 'invoice_id': ID,
        'line_number': Field('positive_integer'), 'supplier_id': ID,
        'contract_id': ID, 'po_id': ID, 'role': ID, 'location': ID,
        'unit': UNIT, 'service_type': SERVICE_TYPE, 'currency': CURRENCY,
        'billed_hours': DECIMAL, 'billed_rate': DECIMAL,
        'billed_amount': DECIMAL, 'invoice_date': DATE, 'description': TEXT,
    },
    'service_entries': {
        'tenant_id': ID, 'entry_id': ID, 'line_id': ID,
        'service_date': DATE, 'approved_hours': DECIMAL, 'approved': BOOL,
    },
    'payments': {
        'tenant_id': ID, 'payment_id': ID, 'line_id': ID,
        'amount': DECIMAL, 'currency': CURRENCY,
        'settled_date': Field('date', optional=True),
        'status': Field('enum', choices=('SETTLED', 'PENDING', 'VOID')),
    },
    'adjustments': {
        'tenant_id': ID, 'adjustment_id': ID, 'line_id': ID,
        'amount': DECIMAL, 'currency': CURRENCY,
        'kind': Field('enum', choices=('APPLIED_CREDIT', 'RECEIVED_REFUND')),
        'status': Field('enum', choices=('APPLIED', 'PENDING', 'VOID')),
        'reason': Field('enum', choices=('RATE_PRICING', 'OTHER')),
        'effective_date': DATE,
    },
    'purchase_orders': {
        'tenant_id': ID, 'po_id': ID, 'supplier_id': ID,
        'contract_id': ID, 'role': ID, 'currency': CURRENCY,
        'unit': UNIT, 'po_rate': DECIMAL,
    },
}

TABLE_KEYS: dict[str, tuple[str, str]] = {
    'suppliers': ('tenant_id', 'supplier_id'),
    'documents': ('tenant_id', 'doc_id'),
    'reviewed_terms': ('tenant_id', 'term_id'),
    'invoice_lines': ('tenant_id', 'line_id'),
    'service_entries': ('tenant_id', 'entry_id'),
    'payments': ('tenant_id', 'payment_id'),
    'adjustments': ('tenant_id', 'adjustment_id'),
    'purchase_orders': ('tenant_id', 'po_id'),
}

_DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')
_NUMBER = re.compile(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')
_TIMESTAMP = re.compile(
    r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}'
    r'(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z'
)


def _error(code: str, file: str, row: int | None, field: str | None,
           message: str) -> dict[str, Any]:
    return {'code': code, 'file': file, 'row': row, 'field': field, 'message': message}


def _valid_date(value: str) -> bool:
    if not _DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _valid_decimal(value: str) -> bool:
    if len(value) > 64 or not _NUMBER.fullmatch(value):
        return False
    try:
        number = Decimal(value)
    except InvalidOperation:
        return False
    if not number.is_finite() or number < Decimal('0'):
        return False
    parts = number.as_tuple()
    return (len(parts.digits) <= 18 and -6 <= parts.exponent <= 12
            and number.adjusted() < 12)


def _valid_path(value: str) -> bool:
    if '\\' in value or '\x00' in value:
        return False
    # Inspect raw components too: PurePosixPath normalizes away a literal '.'.
    pieces = value.split('/')
    if any(piece in ('', '.', '..') or piece.lower().startswith('.env') for piece in pieces):
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and len(pieces) == 2
            and pieces[0] == 'contracts' and path.suffix == '.md'
            and pieces[1] != '.md')


def _field_issue(spec: Field, value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return 'INVALID_TYPE', 'Expected a source string; values are not coerced.'
    if value == '' and spec.optional:
        return None
    if not value.strip():
        return 'REQUIRED_VALUE', 'A nonblank value is required.'
    if spec.kind == 'identifier':
        if value != value.strip() or any(ord(char) < 32 for char in value):
            return 'INVALID_IDENTIFIER', 'Identifiers must have no surrounding whitespace or control characters.'
    elif spec.kind == 'text':
        if '\x00' in value:
            return 'INVALID_TEXT', 'Text must not contain NUL characters.'
    elif spec.kind == 'date':
        if not _valid_date(value):
            return 'INVALID_DATE', 'Expected a valid date in YYYY-MM-DD form.'
    elif spec.kind == 'decimal':
        if not _valid_decimal(value):
            return ('INVALID_DECIMAL', 'Expected a finite nonnegative Decimal string with at most '
                    '18 coefficient digits, six fractional places, and magnitude below 10^12.')
    elif spec.kind == 'currency':
        if not re.fullmatch(r'[A-Z]{3}', value):
            return 'INVALID_CURRENCY', 'Expected a three-letter uppercase currency code.'
    elif spec.kind == 'bool':
        if value not in ('true', 'false'):
            return 'INVALID_BOOLEAN', 'Expected exactly true or false.'
    elif spec.kind == 'enum':
        if value not in spec.choices:
            return 'INVALID_ENUM', 'Value is outside the allowed schema choices.'
    elif spec.kind == 'positive_integer':
        if not re.fullmatch(r'[1-9][0-9]{0,8}', value):
            return 'INVALID_INTEGER', 'Expected a positive integer with at most nine digits.'
    elif spec.kind == 'sha256':
        if not re.fullmatch(r'[0-9a-f]{64}', value):
            return 'INVALID_SHA256', 'Expected a lowercase SHA-256 hexadecimal digest.'
    elif spec.kind == 'path':
        if not _valid_path(value):
            return 'INVALID_PATH', 'Expected a relative contracts/*.md path without traversal or environment-file components.'
    else:
        raise ValueError('Unknown schema field kind')
    return None


def validate_headers(table: str, headers: list[str] | None) -> list[dict]:
    """Reject missing, extra, duplicate, or non-string CSV column names."""
    schema = TABLE_SCHEMAS[table]
    file = f'{table}.csv'
    if headers is None:
        return [_error('MISSING_HEADER', file, 1, None, 'A CSV header row is required.')]
    errors = []
    counts = Counter(header for header in headers if isinstance(header, str))
    for field in schema:
        if field not in counts:
            errors.append(_error('MISSING_COLUMN', file, 1, field, 'Required column is missing.'))
        elif counts[field] > 1:
            errors.append(_error('DUPLICATE_COLUMN', file, 1, field, 'A column name appears more than once.'))
    if any(not isinstance(header, str) or header not in schema for header in headers):
        errors.append(_error('EXTRA_COLUMN', file, 1, None, 'Unexpected columns are not accepted.'))
    if any(count > 1 for header, count in counts.items() if header not in schema):
        errors.append(_error('DUPLICATE_COLUMN', file, 1, None, 'A column name appears more than once.'))
    return errors


def validate_row(table: str, row: dict, row_number: int) -> list[dict]:
    """Validate one CSV row, preserving every raw source value."""
    schema = TABLE_SCHEMAS[table]
    file = f'{table}.csv'
    if not isinstance(row, dict):
        return [_error('INVALID_ROW', file, row_number, None, 'Expected a CSV row mapping.')]
    errors = []
    for field, spec in schema.items():
        if field not in row:
            errors.append(_error('MISSING_FIELD', file, row_number, field, 'Required field is missing.'))
            continue
        issue = _field_issue(spec, row[field])
        if issue:
            errors.append(_error(issue[0], file, row_number, field, issue[1]))
    if any(field not in schema for field in row):
        errors.append(_error('EXTRA_FIELD', file, row_number, None, 'Unexpected fields are not accepted.'))
    if table == 'documents' and row.get('status') == 'SIGNED' and row.get('signed_date') == '':
        errors.append(_error('REQUIRED_VALUE', file, row_number, 'signed_date',
                             'Signed documents require a signed date.'))
    if table == 'payments' and row.get('status') == 'SETTLED' and row.get('settled_date') == '':
        errors.append(_error('REQUIRED_VALUE', file, row_number, 'settled_date',
                             'Settled payments require a settlement date.'))
    if table == 'reviewed_terms':
        reviewer = row.get('reviewer')
        if row.get('reviewed') == 'true' and isinstance(reviewer, str) and not reviewer.strip():
            errors.append(_error('REQUIRED_VALUE', file, row_number, 'reviewer',
                                 'Reviewed terms require a reviewer.'))
        start, end = row.get('effective_from'), row.get('effective_to')
        if (isinstance(start, str) and isinstance(end, str)
                and _valid_date(start) and _valid_date(end) and end < start):
            errors.append(_error('INVALID_DATE_RANGE', file, row_number, 'effective_to',
                                 'The effective end date must not precede the start date.'))
    return errors


def validate_manifest(manifest: Any) -> list[dict]:
    """Validate the complete explicit manifest, including exact boolean flags."""
    file = 'manifest.json'
    if not isinstance(manifest, dict):
        return [_error('INVALID_MANIFEST', file, None, None, 'Expected a JSON object.')]
    errors = []
    required = {
        'synthetic': None, 'dataset_id': ID, 'as_of': DATE,
        'description': TEXT, 'tenants': None,
    }
    for field, spec in required.items():
        if field not in manifest:
            errors.append(_error('MISSING_FIELD', file, None, field, 'Required manifest field is missing.'))
        elif spec:
            issue = _field_issue(spec, manifest[field])
            if issue:
                errors.append(_error(issue[0], file, None, field, issue[1]))
    if manifest.get('synthetic') is not True:
        errors.append(_error('SYNTHETIC_ONLY', file, None, 'synthetic',
                             'Starter accepts synthetic data only. Complete the production gates first.'))
    if any(field not in required and field != 'extracted_at' for field in manifest):
        errors.append(_error('EXTRA_FIELD', file, None, None, 'Unexpected manifest fields are not accepted.'))
    if 'extracted_at' in manifest:
        value = manifest['extracted_at']
        valid = isinstance(value, str) and bool(_TIMESTAMP.fullmatch(value))
        if valid:
            try:
                valid = datetime.fromisoformat(value.replace('Z', '+00:00')).utcoffset() is not None
            except ValueError:
                valid = False
        if not valid:
            errors.append(_error('INVALID_TIMESTAMP', file, None, 'extracted_at',
                                 'Expected an ISO datetime with seconds and an explicit timezone.'))
    if 'tenants' not in manifest:
        return errors
    tenants = manifest['tenants']
    if not isinstance(tenants, dict) or not tenants:
        errors.append(_error('INVALID_TENANTS', file, None, 'tenants',
                             'Expected a nonempty object of tenant attestations.'))
        return errors
    flags = ('contracts_complete', 'payments_complete', 'adjustments_complete')
    for tenant, attestation in tenants.items():
        issue = _field_issue(ID, tenant)
        if issue:
            errors.append(_error(issue[0], file, None, 'tenants', issue[1]))
        prefix = f'tenants.{tenant}' if isinstance(tenant, str) else 'tenants'
        if not isinstance(attestation, dict):
            errors.append(_error('INVALID_ATTESTATION', file, None, prefix,
                                 'Expected a tenant attestation object.'))
            continue
        for field in (*flags, 'attested_by'):
            path = f'{prefix}.{field}'
            if field not in attestation:
                errors.append(_error('MISSING_FIELD', file, None, path, 'Required attestation field is missing.'))
            elif field in flags:
                if type(attestation[field]) is not bool:
                    errors.append(_error('INVALID_BOOLEAN', file, None, path, 'Expected a JSON boolean.'))
            else:
                issue = _field_issue(TEXT, attestation[field])
                if issue:
                    errors.append(_error(issue[0], file, None, path, issue[1]))
        if any(field not in (*flags, 'attested_by') for field in attestation):
            errors.append(_error('EXTRA_FIELD', file, None, prefix, 'Unexpected attestation fields are not accepted.'))
    return errors
