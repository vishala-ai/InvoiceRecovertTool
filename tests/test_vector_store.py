"""Real embedded-Qdrant regressions; embeddings are fixed synthetic feature vectors."""
from __future__ import annotations

import builtins
from datetime import date
import importlib.util
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from recovery.vector_store import LocalVectorStore


HAS_QDRANT = importlib.util.find_spec('qdrant_client') is not None
GENERATION = 'a' * 64
SECOND_GENERATION = 'b' * 64


def chunk(name='one', **changes):
    value = {'chunk_id': str(uuid5(NAMESPACE_URL, 'synthetic-vector-test:' + name)),
             'tenant_id': 'DEMO', 'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
             'role': 'ENGINEER', 'location': 'REMOTE', 'doc_id': 'DOC-A1',
             'source_sha256': '1' * 64, 'status': 'SIGNED', 'kind': 'RATE_AGREEMENT',
             'signed_date': '2026-01-01',
             'signed_date_ordinal': date(2026, 1, 1).toordinal(),
             'text': 'Synthetic signed hourly rate agreement.', 'line_start': 1, 'line_end': 1}
    value.update(changes)
    return value


def filters(**changes):
    value = {'tenant_id': 'DEMO', 'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
             'role': 'ENGINEER', 'location': 'REMOTE', 'status': 'SIGNED',
             'kind': 'RATE_AGREEMENT', 'as_of': '2026-10-15',
             'allowed_doc_ids': ['DOC-A1', 'DOC-A2']}
    value.update(changes)
    return value


class VectorStoreDependencyTests(unittest.TestCase):
    def test_optional_import_is_lazy_and_missing_extra_has_actionable_error(self):
        original = builtins.__import__

        def without_qdrant(name, *args, **kwargs):
            if name == 'qdrant_client' or name.startswith('qdrant_client.'):
                raise ImportError('synthetic dependency unavailable')
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=without_qdrant):
            store = LocalVectorStore('unused-local-vector-path')
            with self.assertRaisesRegex(RuntimeError, 'uv sync --extra vector'):
                with store:
                    self.fail('A missing optional extra cannot open a vector store')

    def test_requires_local_path(self):
        for path in ('', ' ', None, 'https://example.invalid', 'http://localhost:6333'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                LocalVectorStore(path)

    def test_operations_require_context(self):
        store = LocalVectorStore('unused-local-vector-path')
        with self.assertRaisesRegex(RuntimeError, 'with block'):
            store.write_generation(GENERATION, [chunk()], [[1.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, 'with block'):
            store.verify_generation(GENERATION, [chunk()])
        with self.assertRaisesRegex(RuntimeError, 'with block'):
            store.search(GENERATION, [1.0, 0.0], filters())


@unittest.skipUnless(HAS_QDRANT, 'Install optional vector extra to run embedded Qdrant tests')
class LocalVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'vectors'
        self.store = LocalVectorStore(self.path)
        self.store.__enter__()
        self.addCleanup(self.store.__exit__, None, None, None)

    def write_one(self):
        item = chunk()
        self.store.write_generation(GENERATION, [item], [[1.0, 0.0]])
        return item

    def test_real_local_persistence_and_exact_payload(self):
        items = [chunk(), chunk('amendment', doc_id='DOC-A2')]
        vectors = [[1.0, 0.0], [0.8, 0.2]]
        self.store.write_generation(GENERATION, items, vectors)
        self.store.verify_generation(GENERATION, items)
        found = self.store.search(GENERATION, [1.0, 0.0], filters())
        self.assertEqual(len(found), 2)
        expected = {item['chunk_id']: {**item, 'generation': GENERATION} for item in items}
        for result in found:
            score = result.pop('score')
            self.assertTrue(math.isfinite(score))
            self.assertEqual(result, expected[result['chunk_id']])
        self.store.__exit__(None, None, None)
        with LocalVectorStore(self.path) as reopened:
            reopened.verify_generation(GENERATION, items)
            self.assertEqual(len(reopened.search(GENERATION, [1.0, 0.0], filters())), 2)

    def test_client_construction_is_path_only_and_cloud_inference_disabled(self):
        from qdrant_client import QdrantClient
        other = Path(self.temporary.name) / 'other'
        with patch('qdrant_client.QdrantClient', wraps=QdrantClient) as constructor:
            with LocalVectorStore(other):
                constructor.assert_called_once_with(path=str(other), cloud_inference=False)

    def test_context_closes_on_error_and_rejects_double_open(self):
        with self.assertRaisesRegex(RuntimeError, 'already open'):
            self.store.__enter__()
        another = Path(self.temporary.name) / 'error'
        with self.assertRaisesRegex(ValueError, 'test failure'):
            with LocalVectorStore(another):
                raise ValueError('test failure')
        with LocalVectorStore(another) as reopened:
            reopened.write_generation(GENERATION, [chunk()], [[1.0]])

    def test_identical_retry_never_rewrites_existing_collection(self):
        item = chunk()
        self.store.write_generation(GENERATION, [item], [[3.0, -4.0]])
        with patch.object(self.store._client, 'upsert') as upsert, \
             patch.object(self.store._client, 'create_collection') as create, \
             patch.object(self.store._client, 'delete_collection') as delete:
            self.store.write_generation(GENERATION, [item], [[3.0, -4.0]])
        upsert.assert_not_called()
        create.assert_not_called()
        delete.assert_not_called()

    def test_retry_rejects_different_payload_vectors_ids_or_count(self):
        item = self.write_one()
        variants = [([chunk(text='Changed source quotation')], [[1.0, 0.0]]),
                    ([item], [[0.0, 1.0]]),
                    ([chunk('different')], [[1.0, 0.0]]),
                    ([item, chunk('extra')], [[1.0, 0.0], [1.0, 0.0]]),
                    ([item], [[1.0, 0.0, 0.0]])]
        for chunks, vectors in variants:
            with self.subTest(chunks=chunks, vectors=vectors), \
                 patch.object(self.store._client, 'upsert') as upsert, \
                 patch.object(self.store._client, 'delete_collection') as delete:
                with self.assertRaises(RuntimeError):
                    self.store.write_generation(GENERATION, chunks, vectors)
                upsert.assert_not_called()
                delete.assert_not_called()
        self.store.verify_generation(GENERATION, [item])

    def test_shared_collection_prefix_cannot_replace_a_different_full_generation(self):
        item = self.write_one()
        collision = GENERATION[:32] + 'c' * 32
        with self.assertRaisesRegex(RuntimeError, 'payload'):
            self.store.write_generation(collision, [item], [[1.0, 0.0]])
        self.store.verify_generation(GENERATION, [item])

    def test_input_validation_precedes_collection_creation(self):
        invalid = [('', [chunk()], [[1.0]]), ('a' * 63, [chunk()], [[1.0]]),
                   ('g' * 64, [chunk()], [[1.0]]),
                   (GENERATION, [], []), (GENERATION, [chunk()], []),
                   (GENERATION, [chunk()], [[1.0], [1.0]]),
                   (GENERATION, [chunk(), chunk()], [[1.0], [1.0]]),
                   (GENERATION, [chunk(chunk_id='not-a-uuid')], [[1.0]]),
                   (GENERATION, [chunk(source_sha256='bad')], [[1.0]]),
                   (GENERATION, [chunk(tenant_id='')], [[1.0]]),
                   (GENERATION, [chunk(signed_date='2026-02-30')], [[1.0]]),
                   (GENERATION, [chunk(signed_date_ordinal=1)], [[1.0]]),
                   (GENERATION, [chunk(text='')], [[1.0]]),
                   (GENERATION, [chunk(generation=GENERATION)], [[1.0]]),
                   (GENERATION, [chunk(score=0.5)], [[1.0]]),
                   (GENERATION, [chunk(extra=(1, 2))], [[1.0]]),
                   (GENERATION, [chunk(extra=float('nan'))], [[1.0]]),
                   (GENERATION, [chunk()], [[]]), (GENERATION, [chunk()], [[0.0, 0.0]]),
                   (GENERATION, [chunk()], [[True, 1.0]]),
                   (GENERATION, [chunk()], [['1.0', 1.0]]),
                   (GENERATION, [chunk()], [[float('nan'), 1.0]]),
                   (GENERATION, [chunk()], [[float('inf'), 1.0]]),
                   (GENERATION, [chunk(), chunk('two')], [[1.0], [1.0, 0.0]])]
        for generation, chunks, vectors in invalid:
            with self.subTest(generation=generation, chunks=chunks, vectors=vectors):
                with self.assertRaises(ValueError):
                    self.store.write_generation(generation, chunks, vectors)
        self.assertEqual(self.store._client.get_collections().collections, [])

    def test_every_required_source_identity_field_is_mandatory(self):
        for field in ('chunk_id', 'tenant_id', 'supplier_id', 'contract_id', 'role',
                      'location', 'doc_id', 'source_sha256', 'status', 'kind', 'text',
                      'signed_date', 'signed_date_ordinal'):
            item = chunk()
            del item[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.write_generation(GENERATION, [item], [[1.0]])

    def test_extreme_finite_vectors_are_normalized_before_float32_storage(self):
        items = [chunk('tiny'), chunk('large')]
        self.store.write_generation(GENERATION, items, [[1e-200, -1e-200], [1e200, -1e200]])
        self.store.write_generation(GENERATION, items, [[1e-200, -1e-200], [1e200, -1e200]])
        self.store.verify_generation(GENERATION, items)

    def test_verify_scans_all_pages_and_rejects_changed_text(self):
        items = [chunk(str(index)) for index in range(260)]
        self.store.write_generation(GENERATION, items, [[1.0, 0.0] for _ in items])
        self.store.verify_generation(GENERATION, items)
        self.store._client.set_payload(collection_name='recovery_' + GENERATION[:32],
            payload={'text': 'Different source text'}, points=[items[-1]['chunk_id']], wait=True)
        with self.assertRaisesRegex(RuntimeError, 'payload'):
            self.store.verify_generation(GENERATION, items)

    def test_verify_rejects_an_extra_stored_point(self):
        item = self.write_one()
        extra = chunk('unexpected')
        self.store._client.upsert(collection_name='recovery_' + GENERATION[:32],
            points=[self.store._models.PointStruct(id=extra['chunk_id'], vector=[1.0, 0.0],
                payload={**extra, 'generation': GENERATION})], wait=True)
        with self.assertRaisesRegex(RuntimeError, 'count'):
            self.store.verify_generation(GENERATION, [item])

    def test_incompatible_collection_configuration_is_never_replaced(self):
        self.store._client.create_collection(collection_name='recovery_' + GENERATION[:32],
            vectors_config=self.store._models.VectorParams(size=2,
                                                          distance=self.store._models.Distance.DOT))
        with patch.object(self.store._client, 'delete_collection') as delete:
            with self.assertRaisesRegex(RuntimeError, 'configuration'):
                self.store.write_generation(GENERATION, [chunk()], [[1.0, 0.0]])
            delete.assert_not_called()

    def test_interrupted_new_generation_does_not_replace_an_existing_generation(self):
        item = self.write_one()
        original = self.store._client.upsert

        def interrupt(**kwargs):
            original(**{**kwargs, 'points': kwargs['points'][:1]})
            raise RuntimeError('synthetic interrupted staging write')

        pending = [chunk('pending-one'), chunk('pending-two')]
        with patch.object(self.store._client, 'upsert', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.store.write_generation(SECOND_GENERATION, pending, [[1.0, 0.0], [0.0, 1.0]])
        self.store.verify_generation(GENERATION, [item])
        with self.assertRaisesRegex(RuntimeError, 'count'):
            self.store.write_generation(SECOND_GENERATION, pending, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(len(self.store.search(GENERATION, [1.0, 0.0], filters())), 1)

    def test_search_requires_every_scope_guard_and_allowed_id_list(self):
        self.write_one()
        invalid = [filters(status='DRAFT'), filters(kind='NOTE'), filters(tenant_id=''),
                   filters(as_of='20261015'), filters(as_of='2026-02-30'),
                   filters(allowed_doc_ids=None), filters(allowed_doc_ids='DOC-A1'),
                   filters(allowed_doc_ids=['DOC-A1', 'DOC-A1']),
                   filters(allowed_doc_ids=['DOC-A1', 2]), filters(extra='ignored')]
        for key in filters():
            incomplete = filters()
            del incomplete[key]
            invalid.append(incomplete)
        for value in invalid:
            with self.subTest(filters=value), self.assertRaises(ValueError):
                self.store.search(GENERATION, [1.0, 0.0], value)
        for limit in (0, -1, 101, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.store.search(GENERATION, [1.0, 0.0], filters(), limit=limit)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            self.store.search(GENERATION, [1.0], filters())

    def test_empty_allowed_document_set_returns_no_points_without_search(self):
        self.write_one()
        with patch.object(self.store._client, 'query_points') as query:
            self.assertEqual(self.store.search(GENERATION, [1.0, 0.0],
                                              filters(allowed_doc_ids=[])), [])
            query.assert_not_called()

    def test_real_search_filters_scope_signature_kind_as_of_and_allowed_documents(self):
        good = chunk()
        amendment = chunk('amendment', doc_id='DOC-A2')
        excluded = [chunk('wrong-' + field, **{field: 'OTHER'})
                    for field in ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')]
        excluded += [chunk('draft', status='DRAFT'), chunk('note', kind='NOTE'),
                     chunk('not-allowed', doc_id='DOC-NOT-ALLOWED'),
                     chunk('future', signed_date='2026-10-16',
                           signed_date_ordinal=date(2026, 10, 16).toordinal())]
        items = [good, amendment, *excluded]
        self.store.write_generation(GENERATION, items, [[1.0, 0.0] for _ in items])
        self.store.write_generation(SECOND_GENERATION, [chunk('other-generation')], [[1.0, 0.0]])
        found = self.store.search(GENERATION, [1.0, 0.0], filters())
        self.assertEqual({item['chunk_id'] for item in found},
                         {good['chunk_id'], amendment['chunk_id']})
        self.assertEqual(len(self.store.search(GENERATION, [1.0, 0.0], filters(), limit=1)), 1)
        self.assertEqual(self.store.search(GENERATION, [1.0, 0.0],
                                          filters(as_of='2025-12-31')), [])

    def test_unsigned_undated_sources_can_be_indexed_but_cannot_authorize_hits(self):
        draft = chunk('undated', status='DRAFT', signed_date='', signed_date_ordinal=None)
        self.store.write_generation(GENERATION, [draft], [[1.0, 0.0]])
        self.store.verify_generation(GENERATION, [draft])
        self.assertEqual(self.store.search(GENERATION, [1.0, 0.0], filters()), [])

    def test_search_rechecks_untrusted_backend_payload_and_point_identity(self):
        item = self.write_one()
        payload = {**item, 'generation': GENERATION}
        corruptions = [{field: 'OTHER'} for field in
                       ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')]
        corruptions += [{'status': 'DRAFT'}, {'kind': 'NOTE'}, {'doc_id': 'DOC-NOT-ALLOWED'},
                        {'source_sha256': 'bad'}, {'text': ''},
                        {'generation': SECOND_GENERATION}, {'signed_date_ordinal': 1},
                        {'signed_date': '2026-10-16',
                         'signed_date_ordinal': date(2026, 10, 16).toordinal()}]
        for changes in corruptions:
            hit = SimpleNamespace(id=item['chunk_id'], payload={**payload, **changes}, score=0.9)
            with self.subTest(changes=changes), patch.object(self.store._client, 'query_points',
                    return_value=SimpleNamespace(points=[hit])), self.assertRaises(RuntimeError):
                self.store.search(GENERATION, [1.0, 0.0], filters())
        for point_id, score in ((chunk('wrong-id')['chunk_id'], 0.9),
                                (item['chunk_id'], float('nan')), (item['chunk_id'], True)):
            hit = SimpleNamespace(id=point_id, payload=payload, score=score)
            with self.subTest(point_id=point_id, score=score), patch.object(
                    self.store._client, 'query_points', return_value=SimpleNamespace(points=[hit])), \
                    self.assertRaises(RuntimeError):
                self.store.search(GENERATION, [1.0, 0.0], filters())

    def test_missing_generation_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            self.store.verify_generation(GENERATION, [chunk()])
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            self.store.search(GENERATION, [1.0, 0.0], filters())


if __name__ == '__main__':
    unittest.main()
