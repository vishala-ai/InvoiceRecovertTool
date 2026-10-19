"""Optional embedded Qdrant storage for the local synthetic retrieval pilot.

Generations are immutable collections. This adapter has no endpoint, credentials,
server mode, embedding inference, or publication mechanism. The caller publishes
a generation only after verification; an interrupted write may leave an orphan.
Embedded storage is not a production or high-availability service.

Floating-point vectors are similarity features, never financial calculations.
"""
from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path
import re
from uuid import UUID


_HEX = re.compile(r'[0-9a-fA-F]{64}')
_SCOPE = ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')
_IDENTITY = (*_SCOPE, 'doc_id', 'status', 'kind', 'text')
_FILTER_KEYS = {*_SCOPE, 'status', 'kind', 'as_of', 'allowed_doc_ids'}


def _generation(value):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError('Vector generation must be a 64-character hexadecimal digest')
    return value


def _day(value):
    if not isinstance(value, str):
        raise ValueError('Vector evidence dates must use ISO YYYY-MM-DD strings')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('Vector evidence dates must use ISO YYYY-MM-DD strings') from exc
    if parsed.isoformat() != value:
        raise ValueError('Vector evidence dates must use ISO YYYY-MM-DD strings')
    return parsed


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Vector payload must contain finite JSON data') from exc


def _chunks(chunks):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('A vector generation requires a nonempty chunk list')
    result = {}
    for chunk in chunks:
        if not isinstance(chunk, dict) or any(not isinstance(key, str) for key in chunk):
            raise ValueError('Each vector chunk must be a JSON object')
        if 'generation' in chunk or 'score' in chunk:
            raise ValueError('Vector chunks cannot supply reserved result fields')
        chunk_id = chunk.get('chunk_id')
        try:
            canonical_id = str(UUID(chunk_id)) if isinstance(chunk_id, str) else None
        except ValueError as exc:
            raise ValueError('Vector chunk IDs must be canonical UUID strings') from exc
        if canonical_id != chunk_id or canonical_id is None:
            raise ValueError('Vector chunk IDs must be canonical UUID strings')
        if chunk_id in result:
            raise ValueError('Vector chunk IDs must be unique')
        if any(not isinstance(chunk.get(key), str) or not chunk[key].strip()
               for key in _IDENTITY):
            raise ValueError('Vector chunks require complete source and scope identity')
        source_hash = chunk.get('source_sha256')
        if not isinstance(source_hash, str) or not _HEX.fullmatch(source_hash):
            raise ValueError('Vector chunks require a source SHA-256 digest')
        signed_date = chunk.get('signed_date')
        if chunk['status'] == 'SIGNED' or signed_date:
            ordinal = chunk.get('signed_date_ordinal')
            if type(ordinal) is not int or ordinal != _day(signed_date).toordinal():
                raise ValueError('Vector signed-date ordinal must match its source date')
        # Reject values that Qdrant would silently coerce (for example tuple -> list).
        encoded = _json(chunk)
        if json.loads(encoded) != chunk:
            raise ValueError('Vector chunk payload must already use JSON value types')
        result[chunk_id] = json.loads(encoded)
    return result


def _vector(value, dimension=None):
    if not isinstance(value, list) or not value or (dimension is not None
                                                  and len(value) != dimension):
        raise ValueError('Vectors must have a positive, fixed dimension')
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError('Vectors must contain finite numeric components')
    try:
        components = [float(item) for item in value]
    except (OverflowError, ValueError) as exc:
        raise ValueError('Vectors must contain finite numeric components') from exc
    if any(not math.isfinite(item) for item in components):
        raise ValueError('Vectors must contain finite numeric components')
    norm = math.hypot(*components)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Vectors must have a finite, nonzero norm')
    return components


class LocalVectorStore:
    """Context-managed, local-path-only Qdrant store with no automatic inference."""

    def __init__(self, path: str | Path):
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError('A local vector storage path is required')
        if '://' in str(path):
            raise ValueError('Vector storage accepts a local path, not a server URL')
        self.path = Path(path)
        self._client = None
        self._models = None

    def __enter__(self):
        if self._client is not None:
            raise RuntimeError('Local vector store is already open')
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError('Install vector dependencies: uv sync --extra vector') from exc
        self._client = QdrantClient(path=str(self.path), cloud_inference=False)
        self._models = models
        return self

    def __exit__(self, exc_type, exc, traceback):
        client, self._client = self._client, None
        self._models = None
        if client is not None:
            client.close()

    def _opened(self):
        if self._client is None:
            raise RuntimeError('Open the local vector store in a with block first')
        return self._client

    @staticmethod
    def _collection(generation):
        return 'recovery_' + _generation(generation)[:32]

    def _dimension(self, collection):
        client = self._opened()
        if not client.collection_exists(collection_name=collection):
            raise RuntimeError('Vector generation is absent; rebuild the local index')
        params = client.get_collection(collection_name=collection).config.params.vectors
        if (not isinstance(params, self._models.VectorParams) or params.size < 1
                or params.distance != self._models.Distance.COSINE):
            raise RuntimeError('Vector generation has an incompatible collection configuration')
        return params.size

    def _verify(self, generation, expected, vectors=None):
        client = self._opened()
        collection = self._collection(generation)
        dimension = self._dimension(collection)
        if vectors is not None and dimension != len(next(iter(vectors.values()))):
            raise RuntimeError('Existing vector generation has a different vector dimension')
        if client.count(collection_name=collection, exact=True).count != len(expected):
            raise RuntimeError('Vector generation point count does not match canonical evidence')
        seen = set()
        offset = None
        while True:
            records, following = client.scroll(collection_name=collection, limit=256,
                offset=offset, with_payload=True, with_vectors=True)
            for record in records:
                point_id = str(record.id)
                if point_id not in expected or point_id in seen:
                    raise RuntimeError('Vector generation point identities do not match evidence')
                wanted = {**expected[point_id], 'generation': generation}
                if _json(record.payload) != _json(wanted):
                    raise RuntimeError('Vector generation payload does not match canonical evidence')
                actual = _vector(record.vector, dimension)
                if vectors is not None:
                    original = vectors[point_id]
                    norm = math.hypot(*original)
                    # COSINE storage normalizes vectors and persists float32 components.
                    if any(not math.isclose(value, component / norm, rel_tol=2e-6, abs_tol=2e-7)
                           for value, component in zip(actual, original)):
                        raise RuntimeError('Vector generation vectors do not match the requested data')
                seen.add(point_id)
            if following is None:
                break
            if following == offset or not records:
                raise RuntimeError('Vector generation scan did not advance')
            offset = following
        if seen != set(expected):
            raise RuntimeError('Vector generation is missing canonical evidence points')
        if client.count(collection_name=collection, exact=True).count != len(expected):
            raise RuntimeError('Vector generation changed during verification')

    def write_generation(self, generation: str, chunks: list[dict],
                         vectors: list[list[float]]) -> None:
        """Write a new immutable generation; an identical verified retry is harmless.

        Existing conflicting or partially written collections are never replaced.
        COSINE vectors are compared in their normalized float32 representation.
        """
        generation = _generation(generation)
        expected = _chunks(chunks)
        if not isinstance(vectors, list) or len(vectors) != len(expected):
            raise ValueError('Vector and chunk cardinalities must match')
        dimension = len(_vector(vectors[0]))
        supplied = {chunk_id: _vector(vector, dimension)
                    for chunk_id, vector in zip(expected, vectors)}
        client = self._opened()
        collection = self._collection(generation)
        if client.collection_exists(collection_name=collection):
            self._verify(generation, expected, supplied)
            return
        client.create_collection(collection_name=collection,
            vectors_config=self._models.VectorParams(size=dimension,
                                                      distance=self._models.Distance.COSINE))
        # Normalize before float32 storage so very small/large finite feature
        # magnitudes cannot underflow/overflow inside the backend normalizer.
        normalized = {key: [component / math.hypot(*vector) for component in vector]
                      for key, vector in supplied.items()}
        points = [self._models.PointStruct(id=chunk_id, vector=normalized[chunk_id],
                   payload={**chunk, 'generation': generation})
                  for chunk_id, chunk in expected.items()]
        client.upsert(collection_name=collection, points=points, wait=True)
        self._verify(generation, expected, supplied)

    def verify_generation(self, generation: str, chunks: list[dict], *, dimension=None) -> None:
        """Verify every point ID, payload, source text, count and vector shape."""
        if dimension is not None:
            if type(dimension) is not int or dimension < 1:
                raise ValueError('Expected vector dimension must be a positive integer')
            if self._dimension(self._collection(generation)) != dimension:
                raise RuntimeError('Physical vector dimension differs from the pinned index contract')
        self._verify(_generation(generation), _chunks(chunks))

    def search(self, generation: str, query_vector: list[float], filters: dict,
               limit: int = 8) -> list[dict]:
        """Rank only explicitly allowed, signed, as-of-date scoped evidence."""
        generation = _generation(generation)
        query_vector = _vector(query_vector)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Vector search limit must be an integer between 1 and 100')
        if not isinstance(filters, dict) or set(filters) != _FILTER_KEYS:
            raise ValueError('Vector search requires all exact scope and evidence filters')
        if (any(not isinstance(filters[key], str) or not filters[key].strip() for key in _SCOPE)
                or filters['status'] != 'SIGNED' or filters['kind'] != 'RATE_AGREEMENT'):
            raise ValueError('Vector search requires signed rate evidence in one exact scope')
        as_of = _day(filters['as_of']).toordinal()
        allowed = filters['allowed_doc_ids']
        if (not isinstance(allowed, list)
                or any(not isinstance(value, str) or not value.strip() for value in allowed)
                or len(set(allowed)) != len(allowed)):
            raise ValueError('Allowed document IDs must be an explicit unique string list')
        client = self._opened()
        if not allowed:
            return []
        collection = self._collection(generation)
        if self._dimension(collection) != len(query_vector):
            raise ValueError('Query vector dimension does not match the stored generation')
        conditions = [self._models.FieldCondition(key=key,
                       match=self._models.MatchValue(value=filters[key]))
                      for key in (*_SCOPE, 'status', 'kind')]
        conditions.extend([
            self._models.FieldCondition(key='generation',
                match=self._models.MatchValue(value=generation)),
            self._models.FieldCondition(key='doc_id', match=self._models.MatchAny(any=allowed)),
            self._models.FieldCondition(key='signed_date_ordinal',
                range=self._models.Range(lte=as_of)),
        ])
        hits = client.query_points(collection_name=collection, query=query_vector,
            query_filter=self._models.Filter(must=conditions), limit=limit,
            with_payload=True, with_vectors=False).points
        if len(hits) > limit:
            raise RuntimeError('Vector search exceeded its requested result limit')
        result, seen = [], set()
        for hit in hits:
            payload = hit.payload
            if not isinstance(payload, dict) or payload.get('generation') != generation:
                raise RuntimeError('Vector search returned a different evidence generation')
            chunk = {key: value for key, value in payload.items() if key != 'generation'}
            try:
                _chunks([chunk])
            except ValueError as exc:
                raise RuntimeError('Vector search returned invalid source identity') from exc
            if (any(payload[key] != filters[key] for key in (*_SCOPE, 'status', 'kind'))
                    or payload['doc_id'] not in allowed
                    or payload['signed_date_ordinal'] > as_of
                    or str(hit.id) != payload['chunk_id'] or str(hit.id) in seen):
                raise RuntimeError('Vector search returned out-of-scope evidence')
            if (isinstance(hit.score, bool) or not isinstance(hit.score, (int, float))
                    or not math.isfinite(hit.score)):
                raise RuntimeError('Vector search returned an invalid similarity score')
            seen.add(str(hit.id))
            result.append({**payload, 'score': float(hit.score)})
        return result
