"""Serialized, durable synthetic-workspace bundles in private PostgreSQL tables.

This deliberately retains SQLite's financial, trigger and checkpoint semantics.
It is a coarse single-writer pilot bridge, not a relational PostgreSQL migration.
Every app interaction must hold ``workspace`` before accessing any of its files;
callbacks may change session selections only. No database handle may escape it.
The caller must defer success notices until this context has committed, including
catching Streamlit control-flow exceptions *inside* the context and re-raising
them only afterwards. Uncaught application errors discard the local changes.

Install cloud/schema.sql separately using a database administrator. This module
never creates schemas, grants access, imports live accounts or logs credentials.
"""
from __future__ import annotations

from contextlib import closing, contextmanager, suppress
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import sqlite3
import ssl
import tempfile
import threading
import time
from urllib.parse import parse_qsl, urlsplit
import zlib


MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
_FILES = {'portfolio': 'portfolio.sqlite3',
          'checkpoints': 'portfolio-ui-checkpoints.sqlite3',
          'accounts': 'accounts.sqlite3'}
_SCHEMA = 'invoice_recovery_private'
_PROCESS_LOCK = threading.RLock()
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', re.ASCII)
_HASH = re.compile(r'[0-9a-f]{64}', re.ASCII)
_SIDECARS = ('-wal', '-shm', '-journal')
_UNAVAILABLE = 'Cloud workspace is unavailable. Access remains closed; check its configuration.'
_INVALID = 'Cloud workspace failed validation. Access remains closed.'
_UNCERTAIN = ('Save confirmation is unavailable. Refresh to check the current workspace '
              'before retrying the action.')


class CloudStoreError(RuntimeError):
    """Sanitized durable-store failure; never contains DSNs or record payloads."""


def _validate_dsn(dsn):
    try:
        if not isinstance(dsn, str) or not dsn or len(dsn) > 8192:
            raise ValueError
        parsed = urlsplit(dsn)
        host = parsed.hostname
        if (parsed.scheme not in {'postgres', 'postgresql'} or not host
                or not parsed.username or not parsed.password or parsed.fragment
                or not parsed.path or parsed.path == '/'
                or any(character.isspace() for character in dsn)):
            raise ValueError
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
        hostname = host.rstrip('.').lower()
        if hostname == 'localhost' or hostname.endswith('.localhost'):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if (not re.fullmatch(r'[A-Za-z0-9.-]+', host, re.ASCII)
                    or re.fullmatch(r'[0-9.]+', host, re.ASCII)):
                raise ValueError from None
        else:
            mapped = getattr(address, 'ipv4_mapped', None)
            if (address.is_loopback or address.is_unspecified
                    or (mapped is not None and (mapped.is_loopback or mapped.is_unspecified))):
                raise ValueError
        parameters = (parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
                      if parsed.query else [])
        if parameters not in ([], [('sslmode', 'verify-full')]):
            raise ValueError
    except (ValueError, UnicodeError):
        raise CloudStoreError('Choose a hosted PostgreSQL URI; TLS verification is required.') from None


def _validate_certificate(pem):
    try:
        if (not isinstance(pem, str) or not 1 <= len(pem) <= 65536
                or 'PRIVATE KEY' in pem or not pem.lstrip().startswith('-----BEGIN CERTIFICATE-----')):
            raise ValueError
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cadata=pem)
    except (ValueError, ssl.SSLError, UnicodeError):
        raise CloudStoreError('Configure the PostgreSQL server root certificate for TLS verification.') from None


def _regular(path, *, required=False):
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
        raise CloudStoreError(_INVALID)
    if required and not path.is_file():
        raise CloudStoreError('All three workspace databases must exist before publication.')


def _directory(directory):
    directory = Path(directory).absolute()
    if directory == Path(directory.anchor):
        raise CloudStoreError(_INVALID)
    for path in (directory, *directory.parents):
        if path.is_symlink():
            raise CloudStoreError(_INVALID)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not directory.is_dir():
        raise CloudStoreError(_INVALID)
    directory.chmod(0o700)
    return directory


def _checked_paths(directory):
    files = {role: directory / filename for role, filename in _FILES.items()}
    for path in files.values():
        for candidate in (path, *(path.with_name(path.name + suffix) for suffix in _SIDECARS)):
            _regular(candidate)
    return files


def _discard(files):
    # This runs only while the caller owns the process and PostgreSQL locks.
    for path in files.values():
        for candidate in (path, *(path.with_name(path.name + suffix) for suffix in _SIDECARS)):
            _regular(candidate)
            candidate.unlink(missing_ok=True)


def _write_private(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(data)


def _check_sqlite(path):
    try:
        with path.open('rb') as source:
            if source.read(16) != b'SQLite format 3\x00':
                raise CloudStoreError(_INVALID)
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as connection:
            deadline = time.monotonic() + 10
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise CloudStoreError(_INVALID)
    except sqlite3.Error:
        raise CloudStoreError(_INVALID) from None


def _decode(rows):
    decoded = {}
    hashes = {}
    total = 0
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise CloudStoreError(_INVALID)
        role, digest, size, payload = row
        if (not isinstance(role, str) or role not in _FILES or role in decoded
                or not isinstance(digest, str)
                or not _HASH.fullmatch(digest) or type(size) is not int
                or not 100 <= size <= MAX_FILE_BYTES
                or not isinstance(payload, (bytes, bytearray, memoryview))
                or not 1 <= len(payload) <= MAX_COMPRESSED_BYTES):
            raise CloudStoreError(_INVALID)
        total += size
        if total > MAX_TOTAL_BYTES:
            raise CloudStoreError(_INVALID)
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, size + 1)
            if (len(raw) != size or not decoder.eof or decoder.unused_data
                    or decoder.unconsumed_tail or hashlib.sha256(raw).hexdigest() != digest):
                raise CloudStoreError(_INVALID)
        except zlib.error:
            raise CloudStoreError(_INVALID) from None
        decoded[role], hashes[role] = raw, digest
    if set(decoded) != set(_FILES):
        raise CloudStoreError(_INVALID)
    return decoded, hashes


def _restore(files, rows, directory):
    decoded, hashes = _decode(rows)
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=directory) as temporary:
        staged = {}
        for role, raw in decoded.items():
            path = Path(temporary) / _FILES[role]
            _write_private(path, raw)
            _check_sqlite(path)
            staged[role] = path
        # Validate every role before replacing even the first local database.
        _discard(files)
        for role, path in staged.items():
            os.replace(path, files[role])
    return hashes


def _snapshot(files, directory):
    result = {}
    total = 0
    with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=directory) as temporary:
        for role, source in files.items():
            _regular(source, required=True)
            if source.stat().st_size > MAX_FILE_BYTES:
                raise CloudStoreError('Workspace exceeds the synthetic pilot storage limit.')
            for suffix in _SIDECARS:
                _regular(source.with_name(source.name + suffix))
            source.chmod(0o600)
            destination = Path(temporary) / _FILES[role]
            _write_private(destination, b'')
            deadline = time.monotonic() + 10

            page_size = 4096

            def progress(status, remaining, total_pages):
                if time.monotonic() > deadline:
                    raise CloudStoreError('Workspace snapshot timed out. Access remains closed.')
                if total_pages * page_size > MAX_FILE_BYTES:
                    raise CloudStoreError('Workspace exceeds the synthetic pilot storage limit.')

            try:
                # Backup includes committed WAL pages; a raw file copy does not.
                with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=5)) as reader:
                    page_size = reader.execute('PRAGMA page_size').fetchone()[0]
                    with closing(sqlite3.connect(destination)) as writer:
                        reader.backup(writer, pages=256, progress=progress, sleep=0.01)
                        writer.execute('PRAGMA journal_mode=DELETE')
                _check_sqlite(destination)
            except sqlite3.Error:
                raise CloudStoreError(_INVALID) from None
            size = destination.stat().st_size
            total += size
            if not 100 <= size <= MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise CloudStoreError('Workspace exceeds the synthetic pilot storage limit.')
            raw = destination.read_bytes()
            payload = zlib.compress(raw, level=6)
            if len(payload) > MAX_COMPRESSED_BYTES:
                raise CloudStoreError('Workspace exceeds the synthetic pilot storage limit.')
            result[role] = (hashlib.sha256(raw).hexdigest(), size, payload)
    return result


class CloudStore:
    """A private PostgreSQL bundle store; connection details never enter repr/errors."""

    def __init__(self, dsn: str, *, root_cert_pem: str, connector=None):
        _validate_dsn(dsn)
        _validate_certificate(root_cert_pem)
        self._dsn = dsn
        self._root_certificate = root_cert_pem
        self._connector = connector

    @contextmanager
    def workspace(self, workspace_id: str, directory: Path):
        """Restore a locked bundle, run one interaction, and synchronously publish.

        A first workspace yields three absent paths for the caller's explicit
        synthetic initializer. All three valid databases must then be present.
        Failure never falls back to the local files as authoritative storage.
        """
        if not isinstance(workspace_id, str) or not _ID.fullmatch(workspace_id):
            raise CloudStoreError('Choose a workspace identifier using letters, numbers, _ or -.')
        with _PROCESS_LOCK:
            try:
                directory = _directory(directory)
                files = _checked_paths(directory)
            except OSError:
                raise CloudStoreError(_INVALID) from None
            connection = None
            with tempfile.TemporaryDirectory(prefix='recovery-postgres-tls-') as certificate_directory:
                certificate = Path(certificate_directory) / 'root.crt'
                _write_private(certificate, self._root_certificate.encode('ascii'))
                try:
                    connector = self._connector
                    if connector is None:
                        try:
                            import psycopg
                        except ImportError:
                            raise CloudStoreError('Install the cloud extra for PostgreSQL storage.') from None
                        connector = psycopg.connect
                    try:
                        connection = connector(self._dsn, sslmode='verify-full',
                            sslrootcert=str(certificate), connect_timeout=10,
                            prepare_threshold=None, autocommit=False)
                        for setting, value in (('lock_timeout', '10s'),
                                ('statement_timeout', '30s'),
                                ('idle_in_transaction_session_timeout', '120s')):
                            connection.execute(f"SET LOCAL {setting} = '{value}'")
                        connection.execute(f'INSERT INTO {_SCHEMA}.workspace_heads (id,revision) '
                            'VALUES (%s,0) ON CONFLICT (id) DO NOTHING', (workspace_id,))
                        head = connection.execute(f'SELECT revision FROM {_SCHEMA}.workspace_heads '
                            'WHERE id=%s FOR UPDATE', (workspace_id,)).fetchone()
                        if not head or type(head[0]) is not int or head[0] < 0:
                            raise CloudStoreError(_INVALID)
                        revision = head[0]
                        rows = connection.execute(f'SELECT role,sha256,raw_size,payload FROM '
                            f'{_SCHEMA}.workspace_files WHERE workspace_id=%s ORDER BY role',
                            (workspace_id,)).fetchall()
                    except CloudStoreError:
                        raise
                    except Exception:
                        raise CloudStoreError(_UNAVAILABLE) from None
                    try:
                        if revision == 0:
                            if rows:
                                raise CloudStoreError(_INVALID)
                            _discard(files)
                            previous = {}
                        else:
                            previous = _restore(files, rows, directory)
                    except OSError:
                        raise CloudStoreError(_INVALID) from None
                    yield files
                    try:
                        snapshots = _snapshot(files, directory)
                    except OSError:
                        raise CloudStoreError(_INVALID) from None
                    changed = {role: row for role, row in snapshots.items()
                               if previous.get(role) != row[0]}
                    try:
                        for role, (digest, size, payload) in changed.items():
                            connection.execute(f'INSERT INTO {_SCHEMA}.workspace_files '
                                '(workspace_id,role,sha256,raw_size,payload) VALUES (%s,%s,%s,%s,%s) '
                                'ON CONFLICT (workspace_id,role) DO UPDATE SET '
                                'sha256=EXCLUDED.sha256,raw_size=EXCLUDED.raw_size,payload=EXCLUDED.payload',
                                (workspace_id, role, digest, size, payload))
                        if changed:
                            updated = connection.execute(f'UPDATE {_SCHEMA}.workspace_heads '
                                'SET revision=revision+1 WHERE id=%s AND revision=%s RETURNING revision',
                                (workspace_id, revision)).fetchone()
                            if updated != (revision + 1,):
                                raise CloudStoreError(_INVALID)
                    except CloudStoreError:
                        raise
                    except Exception:
                        raise CloudStoreError(_UNCERTAIN) from None
                    try:
                        connection.commit()
                    except Exception:
                        # The server might have committed before the socket failed.
                        raise CloudStoreError(_UNCERTAIN) from None
                except BaseException:
                    if connection is not None:
                        with suppress(Exception):
                            connection.rollback()
                    with suppress(OSError, CloudStoreError):
                        _discard(files)
                    raise
                finally:
                    if connection is not None:
                        with suppress(Exception):
                            connection.close()
