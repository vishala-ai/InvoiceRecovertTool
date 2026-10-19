"""Opt-in synthetic semantic ranking; canonical SQLite evidence stays complete.

The local Qdrant index is a rebuildable projection, never a source of approved
rates, amounts, permissions or amendment completeness. No model is called on a
normal lexical retrieval. An explicitly selected index permits local embeddings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .chunking import CHUNKER_VERSION, chunk_document
from .embeddings import embed_texts, resolve_embedding_config
from .storage import metadata, one_line, scoped_documents
from .vector_store import LocalVectorStore

KIND = 'synthetic-local-qdrant-evidence-v1'
MANIFEST = 'evidence-index.json'
FORMAT = 'plain-source-text-v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _root(path):
    target = Path(path).absolute()
    current = Path(target.anchor)
    for part in target.parts[1:]:
        if part in {'.', '..'} or part.lower().startswith('.env'):
            raise ValueError('Index path must be a separate ordinary local directory')
        current = current / part
        if current.is_symlink():
            raise ValueError('Index paths cannot contain symlinks')
    if target.exists() and not target.is_dir():
        raise ValueError('Index path must be a directory, never a database or source file')
    return target


def _snapshot(conn, tenant):
    manifest = json.loads(metadata(conn, 'manifest'))
    if manifest.get('synthetic') is not True:
        raise ValueError('SYNTHETIC_ONLY: semantic indexing accepts synthetic snapshots only')
    if tenant not in manifest['tenants']:
        raise ValueError('Tenant is not in the synthetic snapshot')
    # Classifying an amendment as a note must not hide it during a new build.
    # The original import reconciliation retains every document field, including
    # classification, which the earlier version table did not store separately.
    from .lifecycle_storage import validate_source_projections
    validate_source_projections(conn)
    all_docs = [dict(row) for row in conn.execute(
        'SELECT * FROM documents WHERE tenant_id=? ORDER BY doc_id', (tenant,))]
    versions = {}
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='source_versions' AND type='table'").fetchone():
        for row in conn.execute('SELECT * FROM source_versions WHERE tenant_id=?', (tenant,)):
            source = dict(row)
            doc = next((doc for doc in all_docs if doc['doc_id'] == source['doc_id']), None)
            scope = json.loads(source['scope_json'])
            if (doc is None or any(doc[key] != value for key, value in scope.items())
                    or any(doc[key] != source[key] for key in ('sha256', 'body', 'status', 'signed_date'))):
                raise ValueError('Source projection disagrees with its immutable version')
            versions[source['doc_id']] = source['version_id']
    documents = []
    for document in all_docs:
        if hashlib.sha256(document['body'].encode('utf-8')).hexdigest() != document['sha256']:
            raise ValueError('Source hash mismatch; semantic index withheld')
        if (document['status'] == 'SIGNED' and document['kind'] == 'RATE_AGREEMENT'
                and document['signed_date'] <= manifest['as_of']):
            document['source_version'] = versions.get(document['doc_id'], 'sha256:' + document['sha256'])
            documents.append(document)
    fingerprint = _digest({'manifest': manifest, 'source_digest': metadata(conn, 'source_digest'),
                           'documents': documents, 'tenant': tenant})
    return manifest, documents, fingerprint


def _chunks(documents, settings):
    if settings.get('version') != CHUNKER_VERSION or set(settings) != {'version', 'max_chars', 'overlap_chars'}:
        raise ValueError('Chunking contract changed; rebuild the semantic index')
    result = []
    for document in documents:
        for chunk in chunk_document(document, max_chars=settings['max_chars'],
                                    overlap_chars=settings['overlap_chars']):
            result.append({**chunk, 'source_version': document['source_version']})
    return result


def _generation(catalog):
    return _digest({key: value for key, value in catalog.items() if key != 'generation'})


def _read(path):
    root = _root(path)
    target = root / MANIFEST
    if target.is_symlink() or not target.is_file() or target.stat().st_size > 262144:
        raise ValueError('A complete published semantic index is required; explicitly build it first')
    from .ollama_provider import strict_json
    try:
        catalog = strict_json(target.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Invalid semantic index manifest; rebuild required') from exc
    fields = {'kind', 'tenant_id', 'source_fingerprint', 'embedding', 'text_format', 'chunking',
              'dimension', 'chunk_count', 'chunks_digest', 'generation', 'build_id'}
    if (not isinstance(catalog, dict) or set(catalog) != fields or catalog['kind'] != KIND
            or catalog['text_format'] != FORMAT or catalog['generation'] != _generation(catalog)
            or type(catalog['dimension']) is not int or not 1 <= catalog['dimension'] <= 4096
            or type(catalog['chunk_count']) is not int or catalog['chunk_count'] <= 0
            or any(not isinstance(catalog[key], str) or not re.fullmatch('[a-f0-9]{64}', catalog[key])
                   for key in ('source_fingerprint', 'chunks_digest', 'generation'))):
        raise ValueError('Invalid or changed semantic index contract; rebuild required')
    embed_texts([], catalog['embedding'])  # Config validation only; no provider call.
    try:
        if str(uuid.UUID(catalog['build_id'])) != catalog['build_id']:
            raise ValueError
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError('Invalid semantic build identity') from exc
    return root, catalog


def resolve_index_config(path):
    """Pin the published generation without contacting a provider or changing data."""
    root, catalog = _read(path)
    return {'index_path': str(root), 'generation': catalog['generation']}


def validate_index_config(conn, tenant, config):
    if not isinstance(config, dict) or set(config) != {'index_path', 'generation'}:
        raise ValueError('A pinned semantic index path and generation are required')
    root, catalog = _read(config['index_path'])
    manifest, documents, fingerprint = _snapshot(conn, tenant)
    if (catalog['generation'] != config['generation'] or catalog['tenant_id'] != tenant
            or fingerprint != catalog['source_fingerprint']):
        raise ValueError('STALE_SEMANTIC_INDEX: sources, scope or index changed; rebuild and start a fresh review')
    chunks = _chunks(documents, catalog['chunking'])
    if len(chunks) != catalog['chunk_count'] or _digest(chunks) != catalog['chunks_digest']:
        raise ValueError('Semantic chunk projection is incomplete or changed; rebuild required')
    return root, catalog, manifest, chunks


def build_index(conn, tenant, path, *, model, base_url='http://127.0.0.1:11434',
                max_chars=1200, overlap_chars=150):
    """Embed complete chunks, reconcile a new generation, then atomically publish."""
    manifest, documents, fingerprint = _snapshot(conn, tenant)
    root = _root(path)
    marker = root / '.recovery-vector-index'
    if root.exists() and any(root.iterdir()) and (
            marker.is_symlink() or not marker.is_file() or marker.read_text() != KIND):
        raise ValueError('Refusing to use an unrelated nonempty directory as a semantic index')
    settings = {'version': CHUNKER_VERSION, 'max_chars': max_chars, 'overlap_chars': overlap_chars}
    chunks = _chunks(documents, settings)
    if not chunks:
        raise ValueError('No signed synthetic rate documents are available to index')
    config = resolve_embedding_config(model, base_url)
    vectors = embed_texts([chunk['text'] for chunk in chunks], config)
    # Detect source changes while embedding, before any new generation is published.
    if _snapshot(conn, tenant)[2] != fingerprint:
        raise ValueError('Sources changed during embedding; index not published')
    catalog = {'kind': KIND, 'build_id': str(uuid.uuid4()),
               'tenant_id': tenant, 'source_fingerprint': fingerprint,
               'embedding': config, 'text_format': FORMAT, 'chunking': settings,
               'dimension': len(vectors[0]), 'chunk_count': len(chunks), 'chunks_digest': _digest(chunks)}
    catalog['generation'] = _generation(catalog)
    root.mkdir(parents=True, exist_ok=True)
    if marker.is_symlink():
        raise ValueError('Index marker must be an ordinary file')
    marker.write_text(KIND)
    with LocalVectorStore(_root(root / 'qdrant')) as store:
        store.write_generation(catalog['generation'], chunks, vectors)
        store.verify_generation(catalog['generation'], chunks, dimension=catalog['dimension'])
    if _snapshot(conn, tenant)[2] != fingerprint:
        raise ValueError('Sources changed during indexing; new generation not published')
    # A failed embedding/store operation leaves the earlier manifest active.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as output:
        temporary = Path(output.name)
        json.dump(catalog, output, sort_keys=True, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    try:
        os.replace(temporary, root / MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)
    return {**catalog, 'document_count': len(documents), 'as_of': manifest['as_of'],
            'index_path': str(root), 'synthetic_only': True}


def rank_evidence(conn, tenant, line_id, documents, config, *, query, limit=8):
    """Rank canonical full documents by chunks, retaining every signed amendment."""
    root, catalog, manifest, chunks = validate_index_config(conn, tenant, config)
    line = one_line(conn, tenant, line_id)
    allowed = {document['doc_id']: document for document in documents}
    canonical_documents = {document['doc_id']: document for document in scoped_documents(conn, line)
                           if document['signed_date'] <= manifest['as_of']}
    if len(documents) != len(allowed) or allowed != canonical_documents:
        raise ValueError('Semantic ranking requires the complete canonical signed document set')
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Semantic hit limit must be an integer from 1 to 100')
    if not isinstance(query, str) or not query.strip() or len(query.encode('utf-8')) > 12000:
        raise ValueError('Semantic query must be nonempty and at most 12000 UTF-8 bytes')
    if not allowed:
        return {'documents': documents, 'matches': [], 'generation': catalog['generation']}
    with LocalVectorStore(_root(root / 'qdrant')) as store:
        store.verify_generation(catalog['generation'], chunks, dimension=catalog['dimension'])
        vector = embed_texts([query], catalog['embedding'])[0]
        if len(vector) != catalog['dimension']:
            raise ValueError('Embedding dimensions changed; rebuild the index')
        hits = store.search(catalog['generation'], vector, {**{key: line[key] for key in
            ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')},
            'status': 'SIGNED', 'kind': 'RATE_AGREEMENT', 'as_of': manifest['as_of'],
            'allowed_doc_ids': list(allowed)}, limit=limit)
    if _snapshot(conn, tenant)[2] != catalog['source_fingerprint']:
        raise ValueError('Sources changed during semantic retrieval; result withheld')
    canonical = {chunk['chunk_id']: chunk for chunk in chunks}
    scores = {}
    matches = []
    for hit in hits:
        chunk = canonical.get(hit.get('chunk_id'))
        if chunk is None or hit['doc_id'] not in allowed or any(
                hit.get(key) != value for key, value in chunk.items()):
            raise ValueError('Vector hit disagrees with canonical source text; result withheld')
        scores[hit['doc_id']] = max(scores.get(hit['doc_id'], float('-inf')), hit['score'])
        matches.append({**chunk, 'score': hit['score']})
    ordered = sorted(documents, key=lambda document: (-scores.get(document['doc_id'], float('-inf')),
                                                      document['doc_id']))
    assert {document['doc_id'] for document in ordered} == set(allowed)
    return {'documents': ordered, 'matches': matches, 'generation': catalog['generation'],
            'complete_signed_document_ids': sorted(allowed),
            'completeness_source': 'canonical_scoped_sqlite_documents',
            'ranking_only': True}
