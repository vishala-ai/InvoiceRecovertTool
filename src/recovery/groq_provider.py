"""Explicit, bounded hosted Groq calls for synthetic explanation drafts only.

The endpoint is fixed, credentials come only from the process environment, and no
SDK, dotenv loading, proxy, redirects, tools, retry or provider fallback is used.
JSON-object mode is compatible with hosted Llama; Python validates the full schema
and exact facts independently in ai.py before a draft can be displayed.
"""
from __future__ import annotations

from http.client import HTTPSConnection
import json
import os
from queue import Empty, Queue
import re
import socket
import threading
import time

from .ollama_provider import (
    CONTEXT_RESERVE,
    DEFAULT_BASE_URL as OLLAMA_BASE_URL,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    OUTPUT_TOKENS,
    strict_json,
)

BASE_URL = 'https://api.groq.com/openai/v1'
DEFAULT_MODEL = 'llama-3.3-70b-versatile'
TIMEOUT_SECONDS = 20
_HOST = 'api.groq.com'
_PATH = '/openai/v1/chat/completions'


def selected_model(model=None) -> str:
    """Validate a model ID without consulting a local secrets file or a provider."""
    selected = model if model is not None else os.environ.get('GROQ_MODEL', DEFAULT_MODEL)
    if (not isinstance(selected, str) or len(selected) > 128
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', selected)
            or '..' in selected or '//' in selected):
        raise ValueError('Select a valid explicit Groq model ID')
    return selected


def hosted_base_url(value=BASE_URL) -> str:
    # ai.py historically defaults to the loopback Ollama origin for every provider.
    # Accept that untouched default, but never an arbitrary external destination.
    if value not in (BASE_URL, BASE_URL + '/', OLLAMA_BASE_URL):
        raise ValueError('Groq uses only its fixed HTTPS API endpoint')
    return BASE_URL


def _api_key() -> str:
    value = os.environ.get('GROQ_API_KEY')
    if (not isinstance(value, str) or not value or len(value) > 4096
            or not value.isascii() or any(character.isspace() or ord(character) < 33
                                          or ord(character) == 127 for character in value)):
        raise ValueError('Set GROQ_API_KEY in the server environment before using Groq')
    return value


def resolve_config(model=None, base_url=BASE_URL) -> tuple[str, str]:
    """Preflight settings without a network request; never return the credential."""
    origin = hosted_base_url(base_url)
    selected = selected_model(model)
    _api_key()
    return selected, origin


class _DeadlineConnection(HTTPSConnection):
    """A cancelled DNS/connect attempt cannot later transmit HTTP credentials."""

    def __init__(self, cancelled, deadline):
        super().__init__(_HOST, timeout=TIMEOUT_SECONDS)
        self.cancelled, self.deadline = cancelled, deadline

    def check_deadline(self):
        remaining = self.deadline - time.monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise TimeoutError
        if self.sock is not None:
            self.sock.settimeout(remaining)

    def connect(self):
        super().connect()
        # DNS resolution itself cannot be interrupted by a Python socket timeout.
        # The caller deadline still returns promptly, and this prevents a late
        # resolver/handshake from sending the request after cancellation.
        try:
            self.check_deadline()
        except TimeoutError:
            self.close()
            raise

    def send(self, data):
        self.check_deadline()
        return super().send(data)


def _abort(connection, cancelled):
    cancelled.set()
    sock = connection.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    connection.close()


def _json_request(payload: dict, api_key: str) -> dict:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError('Full Groq explanation request exceeds the byte limit; draft withheld')
    deadline = time.monotonic() + TIMEOUT_SECONDS
    cancelled = threading.Event()
    connection = _DeadlineConnection(cancelled, deadline)
    outcome = Queue(maxsize=1)

    def request():
        # A daemon worker bounds the caller's total wait, including DNS and TLS.
        # Only safe status categories or bounded bytes cross this boundary.
        result = ('failed', None)
        try:
            connection.request('POST', _PATH, body=data, headers={
                'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json',
                'Accept': 'application/json', 'Accept-Encoding': 'identity',
            })
            connection.check_deadline()
            response = connection.getresponse()
            if response.status == 429:
                result = ('rate_limited', None)
            elif response.status != 200:
                # 3xx is rejected here; no redirect or alternate origin is followed.
                result = ('failed', None)
            elif (response.getheader('Content-Type', '').split(';', 1)[0].strip().lower()
                  != 'application/json'
                  or response.getheader('Content-Encoding', 'identity').lower() != 'identity'):
                result = ('invalid', None)
            else:
                length = response.getheader('Content-Length')
                if length is not None and (not length.isdecimal()
                                           or int(length) > MAX_RESPONSE_BYTES):
                    result = ('oversized', None)
                else:
                    body = bytearray()
                    while True:
                        connection.check_deadline()
                        part = response.read1(min(8192, MAX_RESPONSE_BYTES + 1 - len(body)))
                        if not part:
                            result = ('ok', bytes(body))
                            break
                        body.extend(part)
                        if len(body) > MAX_RESPONSE_BYTES:
                            result = ('oversized', None)
                            break
        except (TimeoutError, socket.timeout):
            result = ('timeout', None)
        except Exception:
            # Provider bodies, network exception messages and credentials must
            # never reach user-facing checkpoint errors or exception chains.
            result = ('failed', None)
        finally:
            connection.close()
            outcome.put(result)

    threading.Thread(target=request, name='groq-explanation', daemon=True).start()
    try:
        category, raw = outcome.get(timeout=max(0, deadline - time.monotonic()))
    except Empty:
        _abort(connection, cancelled)
        raise RuntimeError('Groq request timed out; draft withheld (no provider fallback)') from None
    if category != 'ok':
        messages = {
            'timeout': 'Groq request timed out; draft withheld (no provider fallback)',
            'rate_limited': 'Groq rate limit reached; draft withheld; retry explicitly later',
            'oversized': 'Groq response exceeds the byte limit; draft withheld',
            'invalid': 'Groq returned an invalid response; draft withheld',
        }
        raise RuntimeError(messages.get(category, 'Groq request failed; draft withheld '
                                                 '(no provider fallback)')) from None
    try:
        result = strict_json(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ValueError('Groq returned malformed JSON; draft withheld') from None
    if not isinstance(result, dict) or 'error' in result:
        raise ValueError('Groq returned an invalid response; draft withheld')
    nested = [(result, 0)]
    while nested:
        value, depth = nested.pop()
        if depth > 32:
            raise ValueError('Groq returned an invalid response; draft withheld')
        if isinstance(value, dict):
            nested.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            nested.extend((item, depth + 1) for item in value)
    return result


def generate(*, base_url: str, model: str, context_tokens: int,
             system: str, user: str, schema: dict) -> dict:
    """Send complete evidence once; return only a complete hosted response."""
    model, _ = resolve_config(model=model, base_url=base_url)
    if type(context_tokens) is not int or not 4096 <= context_tokens <= 131072:
        raise ValueError('Groq context_tokens must be an integer from 4096 to 131072')
    if (not isinstance(system, str) or not system or not isinstance(user, str) or not user
            or not isinstance(schema, dict) or not schema):
        raise ValueError('Groq requires a complete prompt and explanation schema')
    payload = {
        'model': model, 'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user}],
        'response_format': {'type': 'json_object'}, 'stream': False,
        'temperature': 0, 'max_completion_tokens': OUTPUT_TOKENS,
    }
    # One UTF-8 byte per token conservatively budgets all evidence and framing.
    # No source is shortened, and API JSON mode is never treated as schema proof.
    request_bytes = len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
    if (request_bytes > MAX_REQUEST_BYTES
            or request_bytes + OUTPUT_TOKENS + CONTEXT_RESERVE > context_tokens):
        raise ValueError('Full evidence, schema and output do not fit the selected context; draft withheld')
    response = _json_request(payload, _api_key())
    choices = response.get('choices')
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    message = choice.get('message') if isinstance(choice, dict) else None
    if (response.get('model') != model or not isinstance(choice, dict)
            or type(choice.get('index')) is not int or choice['index'] != 0
            or choice.get('finish_reason') != 'stop' or not isinstance(message, dict)
            or message.get('role') != 'assistant' or not isinstance(message.get('content'), str)
            or not message['content'].strip() or message.get('refusal') is not None
            or message.get('tool_calls') or message.get('function_call')):
        raise ValueError('Groq generation was incomplete or used an unexpected model; draft withheld')
    if len(message['content'].encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise ValueError('Groq response exceeds the byte limit; draft withheld')
    return response
