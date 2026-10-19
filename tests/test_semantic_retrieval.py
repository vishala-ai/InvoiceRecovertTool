"""Canonical-evidence regressions with actual embedded Qdrant collections.

Only the local embedding provider is simulated. SQLite assessment, chunking,
generation persistence, vector verification and filtered cosine search are real.
"""
from __future__ import annotations

import copy
from contextlib import closing
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recovery import semantic_retrieval as semantic
from recovery.embeddings import _config
from recovery.engine import audit, evaluate_line
from recovery.retrieval import retrieve_evidence
from recovery.storage import connect, load_snapshot
from recovery.vector_store import LocalVectorStore


DATA = Path(__file__).resolve().parents[1] / 'data'
HAS_QDRANT = importlib.util.find_spec('qdrant_client') is not None
EMBEDDING_CONFIG = {'model': 'synthetic-embedding:latest', 'model_digest': 'e' * 64,
                    'base_url': 'http://127.0.0.1:11434'}


@unittest.skipUnless(HAS_QDRANT, 'Install optional vector extra for real semantic retrieval tests')
class SemanticRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.db = self.root / 'synthetic.sqlite3'
        self.index = self.root / 'evidence-index'
        load_snapshot(self.db, DATA)
        context = connect(self.db)
        self.conn = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.embedded_batches = []
        self.current_model_digest = EMBEDDING_CONFIG['model_digest']
        self.query_dimension = 2
        resolver = patch.object(semantic, 'resolve_embedding_config',
                                return_value=copy.deepcopy(EMBEDDING_CONFIG))
        self.config_resolver = resolver.start()
        self.addCleanup(resolver.stop)
        embedder = patch.object(semantic, 'embed_texts', side_effect=self.embed)
        self.embedder = embedder.start()
        self.addCleanup(embedder.stop)

    def embed(self, texts, config):
        selected = _config(config)
        if not texts:
            return []  # Actual adapter validates empty requests without inference.
        if selected['model_digest'] != self.current_model_digest:
            raise ValueError('Embedding model digest changed before inference')
        self.embedded_batches.append(list(texts))
        result = []
        for text in texts:
            # Deliberately rank the outdated agreement above its governing amendment
            # for the default query. This cannot determine which rate is applicable.
            vector = [0.0, 1.0] if '120.00' in text or text == 'prefer-amendment' else [1.0, 0.0]
            if text.startswith('query-dimension-test'):
                vector += [0.0] * (self.query_dimension - len(vector))
            self.assertTrue(all(math.isfinite(value) for value in vector))
            result.append(vector)
        return result

    def build(self, **changes):
        return semantic.build_index(self.conn, 'DEMO', self.index,
                                    model=EMBEDDING_CONFIG['model'], **changes)

    def config(self):
        return semantic.resolve_index_config(self.index)

    def documents(self, tenant='DEMO', line='L001'):
        return retrieve_evidence(self.conn, tenant, line)

    def rank(self, config=None, *, documents=None, tenant='DEMO', line='L001',
             query='prefer-original', limit=8):
        return semantic.rank_evidence(self.conn, tenant, line,
            self.documents(tenant, line) if documents is None else documents,
            self.config() if config is None else config, query=query, limit=limit)

    def dump(self):
        return tuple(self.conn.iterdump())

    def manifest_bytes(self):
        return (self.index / semantic.MANIFEST).read_bytes()

    def change_source(self):
        body = self.conn.execute("SELECT body FROM documents WHERE doc_id='DOC-A2' AND tenant_id='DEMO'").fetchone()[0]
        body += '\nSynthetic source-context revision: no approved financial term changed.\n'
        with self.conn:
            self.conn.execute("UPDATE documents SET body=?,sha256=? WHERE tenant_id='DEMO' AND doc_id='DOC-A2'",
                (body, hashlib.sha256(body.encode('utf-8')).hexdigest()))

    def stored_chunks(self, generation):
        from qdrant_client import QdrantClient
        with closing(QdrantClient(path=str(self.index / 'qdrant'), cloud_inference=False)) as client:
            points, following = client.scroll(collection_name='recovery_' + generation[:32],
                limit=256, with_payload=True, with_vectors=True)
            self.assertIsNone(following)
            return points

    def test_real_build_resolve_retrieve_preserves_original_audit_and_entire_database(self):
        original_dump, original_audit = self.dump(), audit(self.conn)
        built = self.build()
        pinned = self.config()
        found = retrieve_evidence(self.conn, 'DEMO', 'L001', 'prefer-original',
                                  retrieval_config=pinned)
        self.assertEqual({doc['doc_id'] for doc in found}, {'DOC-A1', 'DOC-A2'})
        self.assertEqual(pinned['generation'], built['generation'])
        self.assertEqual(built['embedding'], EMBEDDING_CONFIG)
        self.assertEqual(built['dimension'], 2)
        self.assertEqual(built['document_count'], 7)
        self.assertEqual(built['as_of'], '2026-10-15')
        self.assertTrue(built['synthetic_only'])
        self.assertEqual(audit(self.conn), original_audit)
        self.assertEqual(self.dump(), original_dump)
        self.assertEqual(original_audit['candidate_total'], '80707.50')
        self.assertEqual(original_audit['realized_recovery'], '0.00')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)

    def test_limit_one_and_higher_ranked_outdated_rate_never_drop_governing_amendment(self):
        self.build()
        ranked = self.rank(limit=1)
        self.assertEqual([doc['doc_id'] for doc in ranked['documents']], ['DOC-A1', 'DOC-A2'])
        self.assertEqual(len(ranked['matches']), 1)
        self.assertEqual(ranked['matches'][0]['doc_id'], 'DOC-A1')
        self.assertEqual(ranked['complete_signed_document_ids'], ['DOC-A1', 'DOC-A2'])
        self.assertEqual(ranked['completeness_source'], 'canonical_scoped_sqlite_documents')
        self.assertTrue(ranked['ranking_only'])
        case = evaluate_line(self.conn, 'DEMO', 'L001')
        self.assertEqual(case['net_candidate'], '80000.00')
        self.assertEqual({component['term_id'] for component in case['components']}, {'TERM-A2'})

    def test_actual_zero_hit_search_still_retains_complete_signed_evidence(self):
        from qdrant_client import QdrantClient
        self.build()
        original = QdrantClient.query_points

        def threshold_above_any_cosine(client, *args, **kwargs):
            return original(client, *args, **{**kwargs, 'score_threshold': 1.1})

        # Apply a stricter threshold to the real local search rather than faking
        # the store or its results; there truly are no returned Qdrant points.
        with patch.object(QdrantClient, 'query_points', autospec=True,
                          side_effect=threshold_above_any_cosine):
            ranked = self.rank(query='no lexical or semantic hit', limit=1)
        self.assertEqual(ranked['matches'], [])
        self.assertEqual({doc['doc_id'] for doc in ranked['documents']}, {'DOC-A1', 'DOC-A2'})

    def test_chunk_hits_are_exact_canonical_substrings_with_hash_and_line_provenance(self):
        self.build(max_chars=160, overlap_chars=25)
        ranked = self.rank(query='prefer-amendment', limit=100)
        documents = {doc['doc_id']: doc for doc in ranked['documents']}
        self.assertGreater(len(ranked['matches']), 2)
        for hit in ranked['matches']:
            source = documents[hit['doc_id']]
            start, end = hit['character_start'], hit['character_end']
            self.assertEqual(hit['text'], source['body'][start:end])
            self.assertEqual(hit['source_sha256'], source['sha256'])
            self.assertEqual(hit['line_start'], source['body'][:start].count('\n') + 1)
            self.assertEqual(hit['line_end'], source['body'][:end - 1].count('\n') + 1)
            self.assertTrue(math.isfinite(hit['score']))
            self.assertEqual(hit['tenant_id'], 'DEMO')
            self.assertEqual(hit['status'], 'SIGNED')
            self.assertEqual(hit['kind'], 'RATE_AGREEMENT')
        case = evaluate_line(self.conn, 'DEMO', 'L001')
        for citation in case['citations']:
            self.assertIn(citation['quote'], documents[citation['doc_id']]['body'])
            self.assertEqual(citation['sha256'], documents[citation['doc_id']]['sha256'])

    def test_index_and_search_exclude_other_tenant_unsigned_notes_and_future_sources(self):
        original = dict(self.conn.execute("SELECT * FROM documents WHERE tenant_id='DEMO' AND doc_id='DOC-A2'").fetchone())
        future = {**original, 'doc_id': 'DOC-FUTURE', 'signed_date': '2026-10-16',
                  'path': 'contracts/DOC-FUTURE.md'}
        with self.conn:
            self.conn.execute('INSERT INTO documents (' + ','.join(future) + ') VALUES (' +
                              ','.join('?' for _ in future) + ')', tuple(future.values()))
        before = self.dump()
        built = self.build()
        points = self.stored_chunks(built['generation'])
        self.assertEqual({point.payload['doc_id'] for point in points},
                         {'DOC-A1', 'DOC-A2', 'DOC-B1', 'DOC-B2', 'DOC-C1', 'DOC-C2', 'DOC-C3'})
        ranked = self.rank(limit=100)
        self.assertEqual({match['doc_id'] for match in ranked['matches']}, {'DOC-A1', 'DOC-A2'})
        self.assertEqual(self.dump(), before)
        self.assertEqual(audit(self.conn)['candidate_total'], '80707.50')

    def test_lexical_default_never_resolves_or_invokes_an_embedding_provider(self):
        before = self.dump()
        found = self.documents()
        self.assertEqual({doc['doc_id'] for doc in found}, {'DOC-A1', 'DOC-A2'})
        self.config_resolver.assert_not_called()
        self.embedder.assert_not_called()
        self.assertFalse(self.index.exists())
        self.assertEqual(self.dump(), before)

    def test_missing_index_and_wrong_tenant_fail_before_query_embedding(self):
        before = len(self.embedded_batches)
        with self.assertRaisesRegex(ValueError, 'complete published'):
            semantic.resolve_index_config(self.index)
        self.assertEqual(len(self.embedded_batches), before)
        self.build()
        before = len(self.embedded_batches)
        with self.assertRaisesRegex(ValueError, 'STALE_SEMANTIC_INDEX'):
            semantic.validate_index_config(self.conn, 'OTHER', self.config())
        self.assertEqual(len(self.embedded_batches), before)
        with self.assertRaisesRegex(ValueError, 'Line not found'):
            retrieve_evidence(self.conn, 'OTHER', 'L001', retrieval_config=self.config())
        self.assertEqual(len(self.embedded_batches), before)

    def test_source_revision_or_invalid_hash_fail_before_query_embedding(self):
        self.build()
        config = self.config()
        before = len(self.embedded_batches)
        self.change_source()
        with self.assertRaisesRegex(ValueError, 'STALE_SEMANTIC_INDEX'):
            self.rank(config)
        self.assertEqual(len(self.embedded_batches), before)
        with self.conn:
            self.conn.execute("UPDATE documents SET sha256=? WHERE tenant_id='DEMO' AND doc_id='DOC-A2'",
                              ('0' * 64,))
        with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
            self.rank(config)
        self.assertEqual(len(self.embedded_batches), before)

    def test_dropping_duplicate_or_rewriting_canonical_evidence_is_rejected_before_embedding(self):
        self.build()
        documents = self.documents()
        missing = [doc for doc in documents if doc['doc_id'] != 'DOC-A2']
        rewritten = copy.deepcopy(documents)
        rewritten[0]['body'] = rewritten[0]['body'][10:]
        before = len(self.embedded_batches)
        for supplied in (missing, [*documents, documents[0]], rewritten):
            with self.subTest(doc_ids=[doc['doc_id'] for doc in supplied]):
                with self.assertRaisesRegex(ValueError, 'complete canonical signed document set'):
                    self.rank(documents=supplied)
        self.assertEqual(len(self.embedded_batches), before)

    def test_empty_canonical_scope_returns_no_hits_without_query_embedding(self):
        self.build()
        before = len(self.embedded_batches)
        ranked = self.rank(line='L007')
        self.assertEqual(ranked['documents'], [])
        self.assertEqual(ranked['matches'], [])
        self.assertEqual(len(self.embedded_batches), before)

    def test_corrupt_or_missing_points_fail_before_query_embedding(self):
        from qdrant_client import QdrantClient, models
        built = self.build()
        collection = 'recovery_' + built['generation'][:32]
        points = self.stored_chunks(built['generation'])
        before = len(self.embedded_batches)
        with closing(QdrantClient(path=str(self.index / 'qdrant'), cloud_inference=False)) as client:
            client.set_payload(collection_name=collection, payload={'text': 'Invented substituted quote'},
                               points=[points[0].id], wait=True)
        with self.assertRaisesRegex(RuntimeError, 'payload'):
            self.rank()
        self.assertEqual(len(self.embedded_batches), before)
        with closing(QdrantClient(path=str(self.index / 'qdrant'), cloud_inference=False)) as client:
            client.delete(collection_name=collection,
                          points_selector=models.PointIdsList(points=[points[0].id]), wait=True)
        with self.assertRaisesRegex(RuntimeError, 'count'):
            self.rank()
        self.assertEqual(len(self.embedded_batches), before)

    def test_missing_generation_fails_before_query_embedding(self):
        from qdrant_client import QdrantClient
        built = self.build()
        with closing(QdrantClient(path=str(self.index / 'qdrant'), cloud_inference=False)) as client:
            client.delete_collection(collection_name='recovery_' + built['generation'][:32])
        before = len(self.embedded_batches)
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            self.rank()
        self.assertEqual(len(self.embedded_batches), before)

    def test_physical_collection_dimension_corruption_is_detected_before_query_embedding(self):
        from qdrant_client import QdrantClient, models
        built = self.build()
        collection = 'recovery_' + built['generation'][:32]
        points = self.stored_chunks(built['generation'])
        with closing(QdrantClient(path=str(self.index / 'qdrant'), cloud_inference=False)) as client:
            client.delete_collection(collection_name=collection)
            client.create_collection(collection_name=collection,
                vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
            client.upsert(collection_name=collection, points=[models.PointStruct(id=point.id,
                payload=point.payload, vector=[1.0, 0.0, 0.0]) for point in points], wait=True)
        before = len(self.embedded_batches)
        with self.assertRaisesRegex((RuntimeError, ValueError), 'dimension'):
            self.rank()
        self.assertEqual(len(self.embedded_batches), before,
                         'An incompatible stored generation must fail before query inference')

    def test_model_digest_fault_prevents_query_inference_and_manifest_change(self):
        self.build()
        original = self.manifest_bytes()
        before = len(self.embedded_batches)
        self.current_model_digest = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'model digest changed'):
            self.rank()
        self.assertEqual(len(self.embedded_batches), before)
        self.assertEqual(self.manifest_bytes(), original)

    def test_unexpected_query_embedding_dimension_withholds_result(self):
        self.build()
        original = self.manifest_bytes()
        self.query_dimension = 3
        with self.assertRaisesRegex(ValueError, 'dimensions changed'):
            self.rank(query='query-dimension-test')
        self.assertEqual(self.manifest_bytes(), original)

    def test_failed_embedding_rebuild_preserves_published_manifest_and_old_queries(self):
        self.build()
        original = self.manifest_bytes()
        pinned = self.config()
        with patch.object(semantic, 'embed_texts', side_effect=ValueError('Synthetic embedding failure')):
            with self.assertRaisesRegex(ValueError, 'embedding failure'):
                self.build()
        self.assertEqual(self.manifest_bytes(), original)
        self.assertEqual({doc['doc_id'] for doc in self.rank(pinned)['documents']}, {'DOC-A1', 'DOC-A2'})

    def test_partial_staging_write_keeps_old_manifest_and_fresh_rebuild_uses_new_generation(self):
        first = self.build()
        original, pinned = self.manifest_bytes(), self.config()
        interrupted = []
        write = LocalVectorStore.write_generation

        def partially_write(store, generation, chunks, vectors):
            write(store, generation, chunks[:1], vectors[:1])
            interrupted.append(generation)
            raise RuntimeError('Synthetic interrupted staging generation')

        with patch.object(LocalVectorStore, 'write_generation', new=partially_write):
            with self.assertRaisesRegex(RuntimeError, 'interrupted staging'):
                self.build()
        self.assertEqual(self.manifest_bytes(), original)
        self.assertNotEqual(interrupted[0], first['generation'])
        self.assertEqual(len(self.stored_chunks(interrupted[0])), 1)
        self.assertEqual({doc['doc_id'] for doc in self.rank(pinned)['documents']}, {'DOC-A1', 'DOC-A2'})
        rebuilt = self.build()
        self.assertNotIn(rebuilt['generation'], [first['generation'], interrupted[0]])
        self.assertEqual({doc['doc_id'] for doc in self.rank()['documents']}, {'DOC-A1', 'DOC-A2'})
        with self.assertRaisesRegex(ValueError, 'STALE_SEMANTIC_INDEX'):
            self.rank(pinned)

    def test_same_inputs_explicit_rebuild_changes_generation_and_rejects_old_pinned_config(self):
        first = self.build()
        pinned = self.config()
        rebuilt = self.build()
        self.assertEqual(first['source_fingerprint'], rebuilt['source_fingerprint'])
        self.assertEqual(first['chunks_digest'], rebuilt['chunks_digest'])
        self.assertNotEqual(first['build_id'], rebuilt['build_id'])
        self.assertNotEqual(first['generation'], rebuilt['generation'])
        before = len(self.embedded_batches)
        with self.assertRaisesRegex(ValueError, 'STALE_SEMANTIC_INDEX'):
            self.rank(pinned)
        self.assertEqual(len(self.embedded_batches), before)

    def test_source_change_during_embedding_never_publishes_a_stale_generation(self):
        self.build()
        original = self.manifest_bytes()

        def embed_and_change(texts, config):
            result = self.embed(texts, config)
            if texts:
                self.change_source()
            return result

        with patch.object(semantic, 'embed_texts', side_effect=embed_and_change):
            with self.assertRaisesRegex(ValueError, 'Sources changed during embedding'):
                self.build()
        self.assertEqual(self.manifest_bytes(), original)
        self.assertEqual(audit(self.conn)['candidate_total'], '80707.50')

    def test_synthetic_guard_precedes_provider_and_index_creation(self):
        manifest = json.loads(self.conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        manifest['synthetic'] = False
        with self.conn:
            self.conn.execute("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),))
        with self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
            self.build()
        self.config_resolver.assert_not_called()
        self.embedder.assert_not_called()
        self.assertFalse(self.index.exists())

    def test_reclassified_signed_amendment_cannot_be_hidden_by_a_fresh_index(self):
        from recovery.lifecycle import migrate
        migrate(self.db)
        with self.conn:
            self.conn.execute("UPDATE documents SET kind='NOTE' WHERE tenant_id='DEMO' AND doc_id='DOC-A2'")
        case = evaluate_line(self.conn, 'DEMO', 'L001')
        self.assertEqual(case['status'], 'NEEDS_REVIEW')
        self.assertEqual(case['reason'], 'SOURCE_PROJECTION_MISMATCH')
        with self.assertRaisesRegex(ValueError, 'SOURCE_PROJECTION_MISMATCH'):
            self.build()
        self.config_resolver.assert_not_called()
        self.embedder.assert_not_called()
        self.assertFalse(self.index.exists())

    def test_reclassified_amendment_rejects_existing_index_before_query_inference(self):
        from recovery.lifecycle import migrate
        migrate(self.db)
        self.build()
        pinned = self.config()
        original = self.manifest_bytes()
        before = len(self.embedded_batches)
        with self.conn:
            self.conn.execute("UPDATE documents SET kind='NOTE' WHERE tenant_id='DEMO' AND doc_id='DOC-B2'")
        with self.assertRaisesRegex(ValueError, 'SOURCE_PROJECTION_MISMATCH'):
            self.rank(pinned)
        self.assertEqual(len(self.embedded_batches), before)
        self.assertEqual(self.manifest_bytes(), original)

    def test_unrelated_nonempty_directory_and_invalid_chunking_are_rejected_before_provider(self):
        self.index.mkdir()
        unrelated = self.index / 'keep.txt'
        unrelated.write_text('Existing unrelated contents', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unrelated nonempty directory'):
            self.build()
        self.assertEqual(unrelated.read_text(), 'Existing unrelated contents')
        unrelated.unlink()
        with self.assertRaisesRegex(ValueError, 'max_chars'):
            self.build(max_chars=12)
        self.config_resolver.assert_not_called()
        self.embedder.assert_not_called()

    def test_invalid_queries_and_limits_fail_before_query_inference(self):
        self.build()
        before = len(self.embedded_batches)
        for query in ('', ' ', None, 'é' * 6001):
            with self.subTest(query_type=type(query).__name__):
                with self.assertRaisesRegex(ValueError, 'Semantic query'):
                    self.rank(query=query)
        for limit in (0, True, 101, 1.0):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, 'hit limit'):
                    self.rank(limit=limit)
        self.assertEqual(len(self.embedded_batches), before)


if __name__ == '__main__':
    unittest.main()
