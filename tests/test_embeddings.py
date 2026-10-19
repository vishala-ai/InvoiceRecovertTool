"""Mocked local embedding tests; no HTTP, model downloads or environment loading."""
from __future__ import annotations

import copy
import json
import math
import unittest
from unittest.mock import patch

from recovery import embeddings

MODEL = 'synthetic-embed:latest'
DIGEST = 'a' * 64
CONFIG = {'model': MODEL, 'model_digest': DIGEST, 'base_url': 'http://127.0.0.1:11434'}


def tags(**changes):
    value = {'name': MODEL, 'model': MODEL, 'digest': DIGEST,
             'size': 1000, 'details': {'format': 'gguf'}}
    value.update(changes)
    return {'models': [value]}


def shown(**changes):
    value = {'capabilities': ['embedding'], 'template': ''}
    value.update(changes)
    return value


class EmbeddingTests(unittest.TestCase):
    def network(self, *, tag_response=None, show_response=None, embed_response=None):
        def request(base_url, path, payload=None):
            if path == '/api/tags':
                return tags() if tag_response is None else tag_response
            if path == '/api/show':
                return shown() if show_response is None else show_response
            if path == '/api/embed':
                return ({'model': MODEL, 'embeddings': [[3, 4] for _ in payload['input']]}
                        if embed_response is None else embed_response)
            raise AssertionError('Unexpected endpoint')
        return patch('recovery.ollama_provider._json_request', side_effect=request)

    def test_resolve_pins_exact_installed_model_and_accepts_embedding_only_metadata(self):
        with self.network() as network:
            config = embeddings.resolve_embedding_config(MODEL, 'http://localhost:11434/')
        self.assertEqual(config, CONFIG)
        self.assertEqual([call.args[1] for call in network.call_args_list], ['/api/tags', '/api/show'])
        self.assertEqual(network.call_args.args[2], {'model': MODEL})

    def test_complete_text_batch_truncate_false_normalization_and_inputs_unchanged(self):
        texts = ['Original synthetic rate.\nKeep exact lines.', 'Amendment effective July.']
        before = copy.deepcopy((texts, CONFIG))
        with self.network() as network:
            result = embeddings.embed_texts(texts, CONFIG)
        self.assertEqual(result, [[0.6, 0.8], [0.6, 0.8]])
        self.assertEqual((texts, CONFIG), before)
        requests = [call for call in network.call_args_list if call.args[1] == '/api/embed']
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].args[2], {'model': MODEL, 'input': texts, 'truncate': False})
        self.assertEqual([call.args[1] for call in network.call_args_list],
                         ['/api/tags', '/api/show', '/api/embed', '/api/tags', '/api/show'])

    def test_requests_are_bounded_batches_preserving_input_order(self):
        texts = [str(index) + '\n' + 'a' * 11990 for index in range(5)]
        with self.network() as network:
            result = embeddings.embed_texts(texts, CONFIG)
        payloads = [call.args[2] for call in network.call_args_list if call.args[1] == '/api/embed']
        self.assertEqual([len(payload['input']) for payload in payloads], [2, 2, 1])
        self.assertEqual([text for payload in payloads for text in payload['input']], texts)
        self.assertEqual(len(result), len(texts))
        self.assertTrue(all(len(json.dumps(payload, ensure_ascii=False).encode())
                            <= embeddings.ollama.MAX_REQUEST_BYTES for payload in payloads))

    def test_wire_escaping_is_included_in_request_batch_limit(self):
        texts = ['\x00' * 11000]
        with self.network() as network:
            with self.assertRaisesRegex(ValueError, 'JSON framing'):
                embeddings.embed_texts(texts, CONFIG)
        network.assert_not_called()

    def test_all_input_is_validated_before_any_provider_call(self):
        for texts in [('not a list',), [''], ['   '], [True], ['valid', None],
                      ['valid', 'x' * 12001], ['界' * 4001], ['\ud800']]:
            with self.subTest(texts_type=type(texts)), self.network() as network:
                with self.assertRaises(ValueError):
                    embeddings.embed_texts(texts, CONFIG)
                network.assert_not_called()

    def test_empty_input_is_noop_after_configuration_validation(self):
        with self.network() as network:
            self.assertEqual(embeddings.embed_texts([], CONFIG), [])
        network.assert_not_called()

    def test_invalid_configs_are_rejected_before_any_provider_call(self):
        values = [None, {}, {**CONFIG, 'extra': True}, {**CONFIG, 'model': ''},
                  {**CONFIG, 'model': 'example:cloud'}, {**CONFIG, 'model_digest': 'short'},
                  {**CONFIG, 'base_url': 'https://ollama.com'},
                  {key: value for key, value in CONFIG.items() if key != 'model_digest'}]
        for value in values:
            with self.subTest(config=value), self.network() as network:
                with self.assertRaises(ValueError):
                    embeddings.embed_texts(['synthetic text'], value)
                network.assert_not_called()

    def test_invalid_model_names_and_remote_base_urls_do_not_resolve(self):
        for model in [None, '', ' ', 'synthetic:cloud', 'http://remote/model', 'x' * 257]:
            with self.subTest(model=model), self.network() as network:
                with self.assertRaises(ValueError):
                    embeddings.resolve_embedding_config(model)
                network.assert_not_called()
        with self.network() as network:
            with self.assertRaises(ValueError):
                embeddings.resolve_embedding_config(MODEL, 'http://example.com:11434')
            network.assert_not_called()

    def test_missing_ambiguous_or_nonlocal_installed_metadata_is_rejected(self):
        examples = [{'models': []}, {'models': [tags()['models'][0]] * 2}, {'models': 'invalid'},
                    tags(size=0), tags(size=True), tags(digest='short'), tags(details={'format': 'remote'}),
                    tags(remote_host='https://ollama.com'), tags(remote_model='upstream'),
                    tags(details={'format': 'gguf', 'remote_host': 'https://ollama.com'})]
        for response in examples:
            with self.subTest(response=response), self.network(tag_response=response) as network:
                with self.assertRaises(ValueError):
                    embeddings.resolve_embedding_config(MODEL)
                self.assertFalse(any(c.args[1] == '/api/embed' for c in network.call_args_list))

    def test_show_requires_embedding_capability_and_rejects_remote_aliases(self):
        examples = [shown(capabilities=[]), shown(capabilities=['completion']),
                    shown(capabilities='embedding'), shown(capabilities=['embedding', True]),
                    shown(remote_model='upstream'), shown(remote_host='https://ollama.com')]
        for response in examples:
            with self.subTest(response=response), self.network(show_response=response):
                with self.assertRaises(ValueError):
                    embeddings.resolve_embedding_config(MODEL)

    def test_unexpected_model_remote_metadata_and_error_responses_are_rejected(self):
        for response in [{'model': 'different:latest', 'embeddings': [[1, 0]]},
                         {'model': MODEL, 'embeddings': [[1, 0]], 'remote_host': 'https://ollama.com'},
                         {'model': MODEL, 'embeddings': [[1, 0]], 'remote_model': 'upstream'},
                         {'model': MODEL, 'error': 'failure'}]:
            with self.subTest(response=response), self.network(embed_response=response):
                with self.assertRaises(ValueError):
                    embeddings.embed_texts(['synthetic text'], CONFIG)

    def test_response_cardinality_shape_dimensions_and_coordinates_are_checked(self):
        for vectors in [None, [], [[1], [1]], [[]], ['invalid'], [[1] * 4097],
                        [[True]], [['0.5']], [[None]], [[float('nan')]], [[float('inf')]],
                        [[-float('inf')]], [[10 ** 1000]], [[0, 0]]]:
            with self.subTest(shape=type(vectors)), self.network(
                    embed_response={'model': MODEL, 'embeddings': vectors}):
                with self.assertRaises(ValueError):
                    embeddings.embed_texts(['synthetic text'], CONFIG)

    def test_dimensions_must_be_fixed_within_batch(self):
        with self.network(embed_response={'model': MODEL, 'embeddings': [[1, 0], [1, 0, 0]]}):
            with self.assertRaisesRegex(ValueError, 'dimensions changed'):
                embeddings.embed_texts(['one', 'two'], CONFIG)

    def test_dimensions_must_be_fixed_across_batches(self):
        with self.network() as network:
            original = network.side_effect
            def request(base_url, path, payload=None):
                if path == '/api/embed' and payload['input'] == ['three']:
                    return {'model': MODEL, 'embeddings': [[1, 0, 0]]}
                return original(base_url, path, payload)
            network.side_effect = request
            with self.assertRaisesRegex(ValueError, 'dimensions changed'):
                embeddings.embed_texts(['one', 'two', 'three'], CONFIG)

    def test_extreme_finite_vectors_are_normalized_without_overflow(self):
        with self.network(embed_response={'model': MODEL, 'embeddings': [[1e308] * 4096]}):
            result = embeddings.embed_texts(['synthetic text'], CONFIG)
        self.assertEqual(len(result[0]), 4096)
        self.assertTrue(all(math.isfinite(value) for value in result[0]))
        self.assertAlmostEqual(sum(value * value for value in result[0]), 1)

    def test_pinned_digest_change_blocks_submission(self):
        with self.network(tag_response=tags(digest='b' * 64)) as network:
            with self.assertRaisesRegex(ValueError, 'digest changed'):
                embeddings.embed_texts(['synthetic text'], CONFIG)
        self.assertFalse(any(call.args[1] == '/api/embed' for call in network.call_args_list))

    def test_model_drift_during_request_withholds_result(self):
        with self.network() as network:
            original = network.side_effect
            reads = 0
            def request(base_url, path, payload=None):
                nonlocal reads
                if path == '/api/tags':
                    reads += 1
                    if reads == 2:
                        return tags(digest='b' * 64)
                return original(base_url, path, payload)
            network.side_effect = request
            with self.assertRaisesRegex(ValueError, 'digest changed during'):
                embeddings.embed_texts(['synthetic text'], CONFIG)

    def test_transport_failure_is_propagated_without_any_alternative_provider(self):
        with patch('recovery.ollama_provider._json_request', side_effect=RuntimeError('timed out')) as network:
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                embeddings.embed_texts(['synthetic text'], CONFIG)
        self.assertEqual(network.call_count, 1)


if __name__ == '__main__':
    unittest.main()
