"""Deterministic, lossless character chunks with original-source provenance.

Chunks are search aids, never a completeness guarantee for applicable amendments.
Character offsets address the original Python string, not UTF-8 bytes. Line ranges
include the first and last original characters; CRLF counts as one line break.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date
import hashlib
import json
import re
import uuid


CHUNKER_VERSION = 'source-character-windows-v1'
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, 'urn:supplier-recovery:source-chunks')
_METADATA = ('supplier_id', 'contract_id', 'role', 'location', 'currency', 'unit',
             'service_type', 'status', 'kind', 'signed_date', 'path')


def _document(document):
    if not isinstance(document, dict):
        raise ValueError('Chunking requires a document mapping')
    for field in ('tenant_id', 'doc_id'):
        value = document.get(field)
        if (not isinstance(value, str) or not value.strip() or value != value.strip()
                or any(ord(character) < 32 for character in value)):
            raise ValueError(f'Chunking requires a nonblank {field}')
    body = document.get('body')
    if not isinstance(body, str) or not body.strip():
        raise ValueError('Chunking requires nonempty Unicode source text')
    source_hash = document.get('sha256')
    if not isinstance(source_hash, str) or re.fullmatch('[0-9a-f]{64}', source_hash) is None:
        raise ValueError('Chunking requires a lowercase SHA-256 source hash')
    try:
        actual = hashlib.sha256(body.encode('utf-8')).hexdigest()
    except UnicodeError as exc:
        raise ValueError('Source text must encode as valid UTF-8') from exc
    if actual != source_hash:
        raise ValueError('Source SHA-256 does not match the original Unicode body')
    metadata = {field: document[field] for field in _METADATA if field in document}
    if 'signed_date' in document:
        value = document['signed_date']
        try:
            if not isinstance(value, str):
                raise ValueError('Invalid signed date')
            signed = date.fromisoformat(value)
            if signed.isoformat() != value:
                raise ValueError('Invalid signed date format')
        except ValueError as exc:
            raise ValueError('Signed date must be a calendar date in YYYY-MM-DD form') from exc
        metadata['signed_date_ordinal'] = signed.toordinal()
    return body, source_hash, metadata


def _last_boundary(boundaries, lower, upper):
    index = bisect_right(boundaries, upper) - 1
    return boundaries[index] if index >= 0 and boundaries[index] >= lower else None


def chunk_document(document: dict, *, max_chars=1200, overlap_chars=150) -> list[dict]:
    """Return exact source slices, preferring paragraph and line boundaries.

    A boundary must lie in the latter half of the window, so overlap cannot cause
    a stalled cursor. Long clauses are cut at the character limit. A hard cut is
    moved back one character when needed to keep a CRLF terminator together.
    Requested overlap is retained exactly, including original whitespace.
    """
    if type(max_chars) is not int or not 64 <= max_chars <= 12000:
        raise ValueError('max_chars must be an integer from 64 through 12000')
    if (type(overlap_chars) is not int or overlap_chars < 0
            or overlap_chars * 2 >= max_chars):
        raise ValueError('overlap_chars must be a nonnegative integer below half max_chars')
    body, source_hash, metadata = _document(document)
    breaks = list(re.finditer(r'\r\n|[\r\n]', body))
    line_ends = [match.end() for match in breaks]
    paragraph_ends = [current.end() for previous, current in zip(breaks, breaks[1:])
                      if not body[previous.end():current.start()].strip(' \t')]
    chunks = []
    start = 0
    while start < len(body):
        end = min(start + max_chars, len(body))
        if end < len(body):
            lower = start + (max_chars + 1) // 2
            boundary = (_last_boundary(paragraph_ends, lower, end)
                        or _last_boundary(line_ends, lower, end))
            if boundary is not None:
                end = boundary
            elif body[end - 1:end + 1] == '\r\n':
                end -= 1
        identity = json.dumps([document['tenant_id'], document['doc_id'], source_hash,
            CHUNKER_VERSION, max_chars, overlap_chars, start, end],
            ensure_ascii=False, separators=(',', ':'))
        chunks.append({**metadata, 'chunk_id': str(uuid.uuid5(_NAMESPACE, identity)),
            'tenant_id': document['tenant_id'], 'doc_id': document['doc_id'],
            'source_sha256': source_hash, 'chunker_version': CHUNKER_VERSION,
            'character_start': start, 'character_end': end,
            'line_start': bisect_right(line_ends, start) + 1,
            'line_end': bisect_right(line_ends, end - 1) + 1,
            'text': body[start:end]})
        if end == len(body):
            break
        start = end - overlap_chars
    return chunks
