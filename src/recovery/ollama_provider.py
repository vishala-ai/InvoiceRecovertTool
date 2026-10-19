"""Bounded native HTTP for explicitly selected, installed loopback Ollama models.

The local daemon is trusted infrastructure, not authentication. Its metadata is checked
before sending evidence; a client cannot establish the daemon's server-side cloud policy.
"""
from __future__ import annotations

import json
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

DEFAULT_BASE_URL = 'http://127.0.0.1:11434'
DEFAULT_CONTEXT_TOKENS = 16384
TIMEOUT_SECONDS = 60
MAX_RESPONSE_BYTES = 262144
MAX_REQUEST_BYTES = 60000
OUTPUT_TOKENS = 2048
CONTEXT_RESERVE = 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_base_url(value: str) -> str:
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ValueError('Ollama base URL must be an HTTP loopback origin')
    try:
        url = urlsplit(value)
        port = 11434 if url.port is None else url.port
        if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', '::1', 'localhost'}
                or url.username is not None or url.password is not None
                or url.path not in {'', '/'} or url.query or url.fragment or port < 1):
            raise ValueError
    except ValueError as exc:
        raise ValueError('Ollama base URL must be an HTTP loopback origin') from exc
    # Avoid DNS and proxy configuration, including localhost name-resolution changes.
    host = '[::1]' if url.hostname == '::1' else '127.0.0.1'
    return f'http://{host}:{port}'


def _json_request(base_url: str, path: str, payload: dict | None = None) -> dict:
    origin = local_base_url(base_url)
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    if data is not None and len(data) > MAX_REQUEST_BYTES:
        raise ValueError('Full explanation request exceeds the byte limit; draft withheld')
    request = Request(origin + path, data=data, headers={'Content-Type': 'application/json'})
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError('Ollama request timed out; draft withheld') from exc
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError('Ollama request failed; draft withheld (no provider fallback)') from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError('Ollama response exceeds the byte limit; draft withheld')
    try:
        result = strict_json(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('Ollama returned malformed JSON; draft withheld') from exc
    if not isinstance(result, dict) or 'error' in result:
        raise ValueError('Ollama returned an invalid response; draft withheld')
    return result


def strict_json(value: str | bytes):
    """Reject ambiguous duplicate keys and non-JSON numeric constants."""
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = item
        return result

    def invalid_constant(_value):
        raise ValueError('Nonfinite JSON number')

    return json.loads(value, object_pairs_hook=unique, parse_constant=invalid_constant)


def _reject_remote(value: dict) -> None:
    # Cloud aliases can have ordinary-looking names. Check metadata as well as tags.
    for key, item in value.items():
        if ('remote' in key.lower() or 'cloud' in key.lower()) and item:
            raise ValueError('Only installed local Ollama models are allowed; cloud/remote model rejected')
        if isinstance(item, dict):
            _reject_remote(item)


def installed_model(base_url: str, model: str, context_tokens: int) -> dict:
    if (not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]*', model)
            or '://' in model or 'cloud' in model.lower()):
        raise ValueError('Select an explicit installed local Ollama model')
    if type(context_tokens) is not int or not 4096 <= context_tokens <= 131072:
        raise ValueError('Ollama context_tokens must be an integer from 4096 to 131072')
    response = _json_request(base_url, '/api/tags')
    models = response.get('models')
    if not isinstance(models, list):
        raise ValueError('Ollama returned invalid installed-model metadata')
    matches = [m for m in models if isinstance(m, dict) and m.get('name') == model]
    if len(matches) != 1:
        raise ValueError('Selected Ollama model is not uniquely installed; install it explicitly first')
    selected = matches[0]
    _reject_remote(selected)
    digest, details, size = selected.get('digest'), selected.get('details'), selected.get('size')
    if (not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest)
            or not isinstance(details, dict) or details.get('format') != 'gguf'
            or type(size) is not int or size <= 0):
        raise ValueError('Selected Ollama model lacks verified local GGUF/digest metadata')
    shown = _json_request(base_url, '/api/show', {'model': model})
    _reject_remote(shown)
    info = shown.get('model_info', {})
    architecture = info.get('general.architecture') if isinstance(info, dict) else None
    maximum = info.get(f'{architecture}.context_length') if architecture else None
    if type(maximum) is not int or context_tokens > maximum:
        raise ValueError('Selected context exceeds the model context or its context is unknown')
    template, system, parameters = (shown.get('template'), shown.get('system', ''),
                                     shown.get('parameters', ''))
    if (not isinstance(template, str) or not template.strip()
            or not isinstance(system, str) or not isinstance(parameters, str)):
        raise ValueError('Ollama model prompt template/default system could not be verified')
    overhead_bytes = sum(len(text.encode('utf-8')) for text in (template, system, parameters))
    if overhead_bytes > MAX_REQUEST_BYTES:
        raise ValueError('Ollama model template/default system exceeds the prompt limit')
    return {'model': model, 'model_digest': digest, 'context_tokens': context_tokens,
            'prompt_overhead_bytes': overhead_bytes}


def generate(*, base_url: str, model: str, model_digest: str,
             context_tokens: int, system: str, user: str, schema: dict) -> dict:
    selected = installed_model(base_url, model, context_tokens)
    if selected['model_digest'] != model_digest:
        raise ValueError('Ollama model digest changed; start a new review with the selected model')
    payload = {
        'model': model, 'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user}],
        'format': schema, 'stream': False, 'truncate': False, 'shift': False,
        'options': {'temperature': 0, 'num_ctx': context_tokens, 'num_predict': OUTPUT_TOKENS},
    }
    # One UTF-8 byte per token is a deliberately conservative upper bound, not the
    # usual characters/4 estimate. Include the complete serialized wire request,
    # duplicated schema, model template/default system/parameters, output allowance and
    # framing reserve. Server-side truncate/shift=false additionally reject an expanded
    # template that exceeds context. Never truncate or discard source evidence.
    request_bytes = len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
    needed = request_bytes + selected['prompt_overhead_bytes'] + OUTPUT_TOKENS + CONTEXT_RESERVE
    if request_bytes > MAX_REQUEST_BYTES or needed > context_tokens:
        raise ValueError('Full evidence, schema and output do not fit the selected context; draft withheld')
    response = _json_request(base_url, '/api/chat', payload)
    _reject_remote(response)
    message = response.get('message')
    if (response.get('model') != model or response.get('done') is not True
            or response.get('done_reason') != 'stop' or not isinstance(message, dict)
            or message.get('role') != 'assistant' or not isinstance(message.get('content'), str)
            or message.get('tool_calls')):
        raise ValueError('Ollama generation was incomplete or used an unexpected model; draft withheld')
    prompt_count = response.get('prompt_eval_count')
    if type(prompt_count) is not int or prompt_count <= 0 or prompt_count + OUTPUT_TOKENS > context_tokens:
        raise ValueError('Ollama prompt context could not be verified; draft withheld')
    # Detect a tag change during generation as well as on checkpoint replay.
    if installed_model(base_url, model, context_tokens)['model_digest'] != model_digest:
        raise ValueError('Ollama model digest changed during generation; draft withheld')
    return response
