"""Explicit, bounded local Ollama embeddings. Never used for financial arithmetic.

This adapter neither downloads a model nor selects a cloud provider. The caller owns
consent and must persist the same resolved model identity for indexing and querying.
"""
from __future__ import annotations

import json
import math
import re

from . import ollama_provider as ollama

MAX_TEXT_BYTES = 12000
MAX_DIMENSIONS = 4096
# Keep even high-dimensional responses within the shared bounded HTTP response size.
MAX_BATCH_TEXTS = 2
CONFIG_FIELDS = {'model', 'model_digest', 'base_url'}


def _model_name(value: str) -> str:
    if (not isinstance(value, str) or len(value) > 256
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]*', value)
            or '://' in value or 'cloud' in value.lower()):
        raise ValueError('Select an explicit installed local Ollama embedding model')
    return value


def _installed_embedding_digest(base_url: str, model: str) -> str:
    response = ollama._json_request(base_url, '/api/tags')
    models = response.get('models') if isinstance(response, dict) else None
    if not isinstance(models, list):
        raise ValueError('Ollama returned invalid installed embedding-model metadata')
    matches = [item for item in models if isinstance(item, dict) and item.get('name') == model]
    if len(matches) != 1:
        raise ValueError('Selected embedding model is not uniquely installed; install it explicitly first')
    selected = matches[0]
    ollama._reject_remote(selected)
    digest, details, size = selected.get('digest'), selected.get('details'), selected.get('size')
    if (not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest)
            or not isinstance(details, dict) or details.get('format') != 'gguf'
            or type(size) is not int or size <= 0):
        raise ValueError('Embedding model lacks verified local GGUF/digest metadata')
    shown = ollama._json_request(base_url, '/api/show', {'model': model})
    if not isinstance(shown, dict):
        raise ValueError('Ollama returned invalid embedding capability metadata')
    ollama._reject_remote(shown)
    capabilities = shown.get('capabilities')
    if (not isinstance(capabilities, list) or 'embedding' not in capabilities
            or any(not isinstance(item, str) for item in capabilities)):
        raise ValueError('Selected local model does not advertise embedding capability')
    # Embedding-only models need no chat template or default system prompt.
    return digest


def resolve_embedding_config(model: str, base_url=ollama.DEFAULT_BASE_URL) -> dict:
    """Verify an explicit installed embedding model and pin its local digest."""
    model, base_url = _model_name(model), ollama.local_base_url(base_url)
    return {'model': model, 'model_digest': _installed_embedding_digest(base_url, model),
            'base_url': base_url}


def _config(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise ValueError('Embedding config requires exactly model, model_digest and base_url')
    model = _model_name(value['model'])
    digest = value['model_digest']
    if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('Embedding config requires the pinned 64-character local model digest')
    return {'model': model, 'model_digest': digest,
            'base_url': ollama.local_base_url(value['base_url'])}


def _payload(model: str, texts: list[str]) -> dict:
    return {'model': model, 'input': texts, 'truncate': False}


def _batches(texts: list[str], model: str) -> list[list[str]]:
    if not isinstance(texts, list):
        raise ValueError('Embedding input must be a list of nonempty strings')
    # Validate the entire request before making any provider call, not midway through
    # a partially embedded index. Text is never shortened or silently dropped.
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Embedding input must contain only nonempty strings')
        try:
            size = len(text.encode('utf-8'))
        except UnicodeError:
            raise ValueError('Embedding input must contain valid UTF-8 text') from None
        if size > MAX_TEXT_BYTES:
            raise ValueError('Embedding input exceeds the 12000-byte per-text limit; split it explicitly')
    batches, current = [], []
    for text in texts:
        candidate = current + [text]
        size = len(json.dumps(_payload(model, candidate), ensure_ascii=False).encode('utf-8'))
        if len(candidate) > MAX_BATCH_TEXTS or size > ollama.MAX_REQUEST_BYTES:
            batches.append(current)
            current = [text]
        else:
            current = candidate
        if len(json.dumps(_payload(model, current), ensure_ascii=False).encode('utf-8')) > ollama.MAX_REQUEST_BYTES:
            raise ValueError('Embedding text and JSON framing exceed the request limit')
    if current:
        batches.append(current)
    return batches


def _vectors(response: dict, expected_count: int, dimension: int | None) -> tuple[list[list[float]], int]:
    vectors = response.get('embeddings')
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise ValueError('Ollama embedding response count does not match the input batch')
    normalized = []
    for vector in vectors:
        if not isinstance(vector, list) or not 1 <= len(vector) <= MAX_DIMENSIONS:
            raise ValueError('Ollama embedding dimension must be from 1 to 4096')
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError('Ollama embedding dimensions changed within the request')
        converted = []
        for value in vector:
            if type(value) not in (int, float):
                raise ValueError('Ollama embedding coordinates must be finite numbers, never booleans')
            try:
                coordinate = float(value)
            except (ValueError, OverflowError):
                raise ValueError('Ollama embedding coordinates must be finite numbers') from None
            if not math.isfinite(coordinate):
                raise ValueError('Ollama embedding coordinates must be finite numbers')
            converted.append(coordinate)
        scale = max(abs(value) for value in converted)
        if scale == 0:
            raise ValueError('Ollama returned an unusable zero embedding vector')
        # Scale first so normalization cannot overflow on large finite coordinates.
        scaled = [value / scale for value in converted]
        norm = math.sqrt(sum(value * value for value in scaled))
        normalized.append([value / norm for value in scaled])
    return normalized, dimension


def embed_texts(texts: list[str], config: dict) -> list[list[float]]:
    """Embed complete input text with the same pinned model used by the index."""
    selected = _config(config)
    batches = _batches(texts, selected['model'])
    result, dimension = [], None
    for batch in batches:
        digest = _installed_embedding_digest(selected['base_url'], selected['model'])
        if digest != selected['model_digest']:
            raise ValueError('Embedding model digest changed; rebuild with an explicitly resolved model')
        response = ollama._json_request(selected['base_url'], '/api/embed',
                                        _payload(selected['model'], batch))
        if not isinstance(response, dict):
            raise ValueError('Ollama returned an invalid embedding response')
        ollama._reject_remote(response)
        if response.get('model') != selected['model'] or 'error' in response:
            raise ValueError('Ollama embedding response used an unexpected model or returned an error')
        vectors, dimension = _vectors(response, len(batch), dimension)
        if _installed_embedding_digest(selected['base_url'], selected['model']) != selected['model_digest']:
            raise ValueError('Embedding model digest changed during embedding; result withheld')
        result.extend(vectors)
    return result
