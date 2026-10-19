"""Serialized durable execution for the explicitly selected synthetic Cloud pilot.

The PostgreSQL transaction is the publication boundary. Local SQLite files are
working copies; no rerun/stop is delivered until their coupled state commits.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from . import local_auth
from .groq_provider import DEFAULT_MODEL, selected_model


class CloudRuntimeError(ValueError):
    """An operator-facing error containing no configuration or provider values."""


@dataclass(frozen=True)
class CloudSettings:
    database_url: str = field(repr=False)
    root_cert_pem: str = field(repr=False)
    groq_api_key: str = field(repr=False)
    workspace_id: str = 'invoice-recovery-demo'
    model: str = DEFAULT_MODEL
    owner: Mapping | None = field(default=None, repr=False)


def load_settings(secrets):
    """Read only the explicit Cloud and Groq settings, never dotenv files."""
    try:
        cloud, groq = secrets['cloud'], secrets['groq']
        database, certificate, key = cloud['database_url'], cloud['root_cert_pem'], groq['api_key']
        workspace, model = cloud.get('workspace_id', 'invoice-recovery-demo'), groq.get('model', DEFAULT_MODEL)
        owner = cloud.get('owner')
        if (not all(isinstance(value, str) and value.strip() for value in
                    (database, certificate, key, workspace, model))
                or len(database) > 4096 or len(certificate) > 20000
                or not 16 <= len(key) <= 512 or any(c.isspace() for c in key)
                or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,63}', workspace)
                or any(value.startswith(('SET_', 'REPLACE_', 'PASTE_')) for value in (key,))
                or 'YOUR_SESSION_POOLER_HOST' in database):
            raise ValueError
        if owner is not None and not isinstance(owner, Mapping):
            raise ValueError
        return CloudSettings(database, certificate, key, workspace, selected_model(model), owner)
    except Exception:
        raise CloudRuntimeError('Complete the private cloud and groq sections in Streamlit Secrets. '
                                'Credentials must never be committed to GitHub.') from None


@lru_cache(maxsize=16)
def _runtime_directory(workspace, connection_identity):
    # Stable across reruns so invoice selection/widget identities stay stable.
    return Path(tempfile.mkdtemp(prefix='invoice-recovery-cloud-')).resolve()


def runtime_directory(settings):
    identity = hashlib.sha256(settings.database_url.encode()).hexdigest()
    return _runtime_directory(settings.workspace_id, identity)


@contextmanager
def cloud_environment(settings, files):
    values = {'RECOVERY_CLOUD': '1', 'RECOVERY_DB': str(files['portfolio']),
              'RECOVERY_AUTH_DB': str(files['accounts']), 'GROQ_API_KEY': settings.groq_api_key,
              'GROQ_MODEL': settings.model, 'PYTHON_DOTENV_DISABLED': '1',
              'RECOVERY_RETRIEVAL_INDEX': None}
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_and_provision(settings, files):
    from .storage import connect, metadata
    from .lifecycle_storage import validate_source_projections

    if not all(path.is_file() for path in files.values()):
        raise CloudRuntimeError('The Cloud workspace needs its initial synthetic seed. '
                                'The operator must run scripts/seed_cloud_workspace.py first.')
    with connect(files['portfolio']) as conn:
        manifest = json.loads(metadata(conn, 'manifest'))
        if manifest.get('synthetic') is not True:
            raise CloudRuntimeError('Only the validated synthetic demonstration is permitted in Cloud.')
        validate_source_projections(conn)
    local_auth.initialize(files['accounts'])
    if not local_auth.has_accounts(files['accounts']):
        owner = settings.owner
        if not owner:
            raise CloudRuntimeError('The operator must provision the first Cloud owner through '
                                    'the private cloud.owner settings. Public registration is disabled.')
        try:
            password = owner['password']
            if not isinstance(password, str) or password.startswith(('REPLACE_', 'SET_', 'PASTE_')):
                raise ValueError
            local_auth.bootstrap_owner(files['accounts'], owner['username'], owner['display_name'], password)
        except (KeyError, TypeError, ValueError, PermissionError):
            raise CloudRuntimeError('Cloud owner provisioning failed. Check the private owner '
                                    'settings and use a unique 15–128 character passphrase.') from None


def clear_session(state):
    for key in list(state):
        del state[key]


def execute_workspace(settings, state, runner, *, store_factory=None, directory=None,
                      control_exceptions=()):
    """Commit before honoring Streamlit control flow; fail closed on save ambiguity.

    The caller must render a fresh error page after failure, clearing any UI
    emitted provisionally before a database connection/commit was lost.
    """
    from .cloud_store import CloudStore, CloudStoreError

    store_factory = store_factory or CloudStore
    directory = Path(directory).resolve() if directory is not None else runtime_directory(settings)
    pending = None
    try:
        store = store_factory(settings.database_url, root_cert_pem=settings.root_cert_pem)
        with store.workspace(settings.workspace_id, directory) as files:
            with cloud_environment(settings, files):
                _validate_and_provision(settings, files)
                try:
                    runner()
                except control_exceptions as control:
                    pending = control
    except CloudRuntimeError:
        clear_session(state)
        raise
    except CloudStoreError:
        clear_session(state)
        raise CloudRuntimeError('Save confirmation is unavailable. Refresh to check the current '
                                'saved state before retrying an action.') from None
    except Exception:
        clear_session(state)
        raise CloudRuntimeError('The Cloud workspace could not complete this request. '
                                'Refresh to check the saved state; contact the operator if it persists.') from None
    if pending is not None:
        raise pending
