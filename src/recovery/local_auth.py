"""Password accounts for the localhost synthetic workspace, separate from finance data.

This is a local access gate, not tenant isolation or a production identity provider.
There are no default accounts. ``bootstrap_owner`` is first-run-only; additional
accounts are created by that owner. Public account dictionaries contain only
username, display_name, role, active and created_at. Passwords and opaque session
tokens must never be logged or put into URLs/cookies; the UI retains a token only
in server-side Streamlit session state.

Password storage uses hashlib.scrypt's OWASP-listed N=2**15, r=8, p=3 option
(32 MiB), a unique 16-byte salt and a 32-byte key. Sessions expire after 30 idle
minutes or eight hours, whichever comes first. Persistent attempt reservations
limit known and unknown names alike to five per 15 minutes and the whole store
to 50 per 15 minutes. Successful login clears only that username's counter.

Malformed account fields raise ValueError; missing authority raises
PermissionError. Authentication failures (including throttling) return None.
Storage failures raise AuthError without including credentials or SQL values.
All state lives in the explicitly initialized, private authentication database.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time


MIN_PASSWORD_LENGTH = 15
MAX_PASSWORD_LENGTH = 128
IDLE_TIMEOUT_SECONDS = 30 * 60
ABSOLUTE_TIMEOUT_SECONDS = 8 * 60 * 60
ATTEMPT_WINDOW_SECONDS = 15 * 60
USERNAME_ATTEMPT_LIMIT = 5
GLOBAL_ATTEMPT_LIMIT = 50
MAX_SESSIONS_PER_ACCOUNT = 5
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**15, 8, 3
_HASH_SCHEME = 'scrypt-n32768-r8-p3-dk32-v1'
_TABLES = {'auth_meta', 'auth_accounts', 'auth_sessions', 'auth_throttles'}
_USER_PATTERN = re.compile(r'[a-z0-9][a-z0-9._-]{2,63}', re.ASCII)
_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9_-]{43}', re.ASCII)
_SCHEMA = (
    'CREATE TABLE auth_meta (schema_version INTEGER PRIMARY KEY CHECK(schema_version=1))',
    '''CREATE TABLE auth_accounts (
        username TEXT PRIMARY KEY, display_name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('owner','accountant')),
        active INTEGER NOT NULL CHECK(active IN (0,1)),
        salt BLOB NOT NULL, password_hash BLOB NOT NULL, hash_scheme TEXT NOT NULL,
        security_version INTEGER NOT NULL, created_at INTEGER NOT NULL)''',
    '''CREATE TABLE auth_sessions (
        token_hash TEXT PRIMARY KEY,
        username TEXT NOT NULL REFERENCES auth_accounts(username),
        security_version INTEGER NOT NULL, created_at INTEGER NOT NULL,
        last_seen_at INTEGER NOT NULL)''',
    '''CREATE TABLE auth_throttles (
        bucket_key TEXT PRIMARY KEY, window_start INTEGER NOT NULL,
        attempts INTEGER NOT NULL)''',
    'CREATE INDEX auth_sessions_user ON auth_sessions(username)',
)


class AuthError(ValueError):
    """The separate local authentication store is unavailable or incompatible."""


def _now() -> int:
    return int(time.time())


def _username(value: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError('Use a username of 3–64 letters, numbers, dots, underscores or hyphens.')
    value = value.strip().lower()
    if not _USER_PATTERN.fullmatch(value):
        raise ValueError('Use a username of 3–64 letters, numbers, dots, underscores or hyphens.')
    return value


def _display_name(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 80:
        raise ValueError('Enter a display name of 1–80 characters.')
    value = value.strip()
    if not value.isprintable():
        raise ValueError('Display names cannot contain control characters.')
    return value


def _password(value: str) -> bytes:
    if (not isinstance(value, str) or
            not MIN_PASSWORD_LENGTH <= len(value) <= MAX_PASSWORD_LENGTH or not value.strip()):
        raise ValueError('Use a passphrase of 15–128 characters.')
    try:
        encoded = value.encode('utf-8')
    except UnicodeError:
        raise ValueError('Use a valid Unicode passphrase.') from None
    if len(encoded) > 512:
        raise ValueError('Use a passphrase of at most 512 UTF-8 bytes.')
    return encoded


def _derive(password: bytes, salt: bytes) -> bytes:
    if not hasattr(hashlib, 'scrypt'):
        raise AuthError('Local password hashing requires OpenSSL scrypt; start the app with uv.')
    try:
        return hashlib.scrypt(password, salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
                              maxmem=64 * 1024 * 1024, dklen=32)
    except (ValueError, MemoryError):
        raise AuthError('Local password hashing is unavailable; sign-in was not completed.') from None


def _credentials(password: str) -> tuple[bytes, bytes]:
    encoded = _password(password)
    salt = secrets.token_bytes(16)
    return salt, _derive(encoded, salt)


def _public(row: sqlite3.Row) -> dict:
    return {'username': row['username'], 'display_name': row['display_name'],
            'role': row['role'], 'active': bool(row['active']),
            'created_at': datetime.fromtimestamp(row['created_at'], timezone.utc).isoformat()}


def _checked_path(db_path: str | Path, *, create: bool = False) -> Path:
    path = Path(db_path).expanduser().absolute()
    if path.name in {'', '.', '..'}:
        raise AuthError('Choose a separate local account database file.')
    parent = path.parent
    if create:
        # Only new directories receive private permissions; existing workspace
        # directories are never chmodded. Do not follow a supplied parent link.
        missing = []
        cursor = parent
        while not cursor.exists() and not cursor.is_symlink():
            missing.append(cursor)
            cursor = cursor.parent
        if cursor.is_symlink():
            raise AuthError('The local account directory cannot be a symbolic link.')
        for directory in reversed(missing):
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                pass  # Another first-run session may create this directory.
    try:
        parent_info = parent.lstat()
        if (not stat.S_ISDIR(parent_info.st_mode) or parent.is_symlink() or
                parent_info.st_mode & 0o022):
            raise AuthError('The local account directory must not be writable by other users.')
        if hasattr(os, 'getuid') and parent_info.st_uid != os.getuid():
            raise AuthError('The local account directory must belong to the current OS user.')
        if create:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                                     getattr(os, 'O_NOFOLLOW', 0), 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077 or
                (hasattr(os, 'getuid') and info.st_uid != os.getuid())):
            raise AuthError('The local account file must be a private regular file (mode 0600).')
        for suffix in ('-journal', '-wal', '-shm'):
            sibling = Path(f'{path}{suffix}')
            if sibling.is_symlink():
                raise AuthError('Local account journal files cannot be symbolic links.')
    except OSError:
        raise AuthError('The local account store is unavailable.') from None
    return path


@contextmanager
def _connection(db_path: str | Path, *, create: bool = False):
    path = _checked_path(db_path, create=create)
    conn = None
    try:
        conn = sqlite3.connect(path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA trusted_schema=OFF')
        yield conn
    except sqlite3.Error:
        raise AuthError('The local account store is unavailable or incompatible.') from None
    finally:
        if conn is not None:
            conn.close()


def _check_schema(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables != _TABLES or [row[0] for row in conn.execute(
            'SELECT schema_version FROM auth_meta')] != [1]:
        raise AuthError('Use a separate, initialized local account database.')


@contextmanager
def _transaction(db_path: str | Path):
    with _connection(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            _check_schema(conn)
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def initialize(db_path: str | Path) -> None:
    """Create only the separate account schema; refuse an existing business DB."""
    with _connection(db_path, create=True) as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables:
                for statement in _SCHEMA:
                    conn.execute(statement)
                conn.execute('INSERT INTO auth_meta VALUES (1)')
            _check_schema(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def has_accounts(db_path: str | Path) -> bool:
    """An empty account store permits setup; disabled accounts never reopen it."""
    with _transaction(db_path) as conn:
        return conn.execute('SELECT 1 FROM auth_accounts LIMIT 1').fetchone() is not None


def bootstrap_owner(db_path: str | Path, username: str, display_name: str, password: str) -> dict:
    """Atomically create the first owner; creation does not sign the user in."""
    username, display_name = _username(username), _display_name(display_name)
    _password(password)
    if has_accounts(db_path):
        raise PermissionError('Initial account setup has already been completed.')
    salt, password_hash = _credentials(password)
    with _transaction(db_path) as conn:
        if conn.execute('SELECT 1 FROM auth_accounts LIMIT 1').fetchone():
            raise PermissionError('Initial account setup has already been completed.')
        conn.execute('INSERT INTO auth_accounts VALUES (?,?,?,?,?,?,?,?,?)',
                     (username, display_name, 'owner', 1, salt, password_hash,
                      _HASH_SCHEME, 1, _now()))
        return _public(conn.execute('SELECT * FROM auth_accounts WHERE username=?',
                                    (username,)).fetchone())


def _bucket(username: str) -> str:
    return 'user:' + hashlib.sha256(username.encode('ascii')).hexdigest()


def _reserve_attempt(conn: sqlite3.Connection, username: str, now: int) -> bool:
    conn.execute('DELETE FROM auth_throttles WHERE window_start<=?',
                 (now - ATTEMPT_WINDOW_SECONDS,))
    for key, limit in (('global', GLOBAL_ATTEMPT_LIMIT), (_bucket(username), USERNAME_ATTEMPT_LIMIT)):
        row = conn.execute('SELECT * FROM auth_throttles WHERE bucket_key=?', (key,)).fetchone()
        if row and (row['window_start'] > now or row['attempts'] >= limit):
            return False
    for key in ('global', _bucket(username)):
        conn.execute('''INSERT INTO auth_throttles VALUES (?,?,1)
            ON CONFLICT(bucket_key) DO UPDATE SET attempts=attempts+1''', (key, now))
    return True


def _matches(encoded: bytes, row: sqlite3.Row | None) -> bool:
    # Unknown users still incur the same scrypt cost. No placeholder account or
    # secret exists, and no supplied value is stored in the attempt counters.
    usable = (row is not None and row['hash_scheme'] == _HASH_SCHEME and
              isinstance(row['salt'], bytes) and len(row['salt']) == 16 and
              isinstance(row['password_hash'], bytes) and len(row['password_hash']) == 32)
    salt = row['salt'] if usable else bytes(16)
    expected = row['password_hash'] if usable else bytes(32)
    equal = hmac.compare_digest(_derive(encoded, salt), expected)
    return bool(usable and equal and row['active'])


def authenticate(db_path: str | Path, username: str, password: str) -> str | None:
    """Return a fresh opaque token, or the same None for every sign-in failure."""
    try:
        username, encoded = _username(username), _password(password)
    except ValueError:
        return None
    now = _now()
    with _transaction(db_path) as conn:
        if not _reserve_attempt(conn, username, now):
            return None
        row = conn.execute('SELECT * FROM auth_accounts WHERE username=?', (username,)).fetchone()
    if not _matches(encoded, row):
        return None
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode('ascii')).hexdigest()
    with _transaction(db_path) as conn:
        fresh = conn.execute('SELECT * FROM auth_accounts WHERE username=?', (username,)).fetchone()
        if (not fresh or not fresh['active'] or fresh['security_version'] != row['security_version'] or
                not hmac.compare_digest(fresh['password_hash'], row['password_hash'])):
            return None
        now = _now()
        conn.execute('DELETE FROM auth_sessions WHERE created_at<=? OR last_seen_at<=?',
                     (now - ABSOLUTE_TIMEOUT_SECONDS, now - IDLE_TIMEOUT_SECONDS))
        conn.execute('INSERT INTO auth_sessions VALUES (?,?,?,?,?)',
                     (token_hash, username, fresh['security_version'], now, now))
        conn.execute('''DELETE FROM auth_sessions WHERE username=? AND token_hash NOT IN
            (SELECT token_hash FROM auth_sessions WHERE username=?
             ORDER BY created_at DESC, rowid DESC LIMIT ?)''',
                     (username, username, MAX_SESSIONS_PER_ACCOUNT))
        conn.execute('DELETE FROM auth_throttles WHERE bucket_key=?', (_bucket(username),))
    return token


def _token_hash(token: str) -> str | None:
    if not isinstance(token, str) or not _TOKEN_PATTERN.fullmatch(token):
        return None
    return hashlib.sha256(token.encode('ascii')).hexdigest()


def _session(conn: sqlite3.Connection, token: str, now: int) -> sqlite3.Row | None:
    digest = _token_hash(token)
    if digest is None:
        return None
    row = conn.execute('''SELECT a.*,s.created_at AS session_created_at,s.last_seen_at,
        s.security_version AS session_version FROM auth_sessions s
        JOIN auth_accounts a ON a.username=s.username WHERE s.token_hash=?''', (digest,)).fetchone()
    if row is None:
        return None
    if (not row['active'] or row['security_version'] != row['session_version'] or
            now < row['session_created_at'] or now < row['last_seen_at'] or
            now - row['session_created_at'] >= ABSOLUTE_TIMEOUT_SECONDS or
            now - row['last_seen_at'] >= IDLE_TIMEOUT_SECONDS):
        conn.execute('DELETE FROM auth_sessions WHERE token_hash=?', (digest,))
        return None
    conn.execute('UPDATE auth_sessions SET last_seen_at=? WHERE token_hash=?', (now, digest))
    return row


def _require_user(conn: sqlite3.Connection, token: str) -> sqlite3.Row:
    row = _session(conn, token, _now())
    if row is None:
        raise PermissionError('Sign in again to continue.')
    return row


def _require_owner(conn: sqlite3.Connection, token: str) -> sqlite3.Row:
    row = _require_user(conn, token)
    if row['role'] != 'owner':
        raise PermissionError('Only the workspace owner can manage local accounts.')
    return row


def current_user(db_path: str | Path, token: str) -> dict | None:
    """Fresh server-side token validation, including active state and expiry."""
    if _token_hash(token) is None:
        return None
    with _transaction(db_path) as conn:
        row = _session(conn, token, _now())
        return _public(row) if row is not None else None


def logout(db_path: str | Path, token: str) -> None:
    digest = _token_hash(token)
    if digest is None:
        return
    with _transaction(db_path) as conn:
        conn.execute('DELETE FROM auth_sessions WHERE token_hash=?', (digest,))


def create_account(db_path: str | Path, actor_token: str, username: str,
                   display_name: str, password: str) -> dict:
    """An active owner may add an accountant, never another owner."""
    with _transaction(db_path) as conn:
        _require_owner(conn, actor_token)
    username, display_name = _username(username), _display_name(display_name)
    salt, password_hash = _credentials(password)
    with _transaction(db_path) as conn:
        _require_owner(conn, actor_token)
        if conn.execute('SELECT 1 FROM auth_accounts WHERE username=?', (username,)).fetchone():
            raise ValueError('Choose a different username.')
        conn.execute('INSERT INTO auth_accounts VALUES (?,?,?,?,?,?,?,?,?)',
                     (username, display_name, 'accountant', 1, salt, password_hash,
                      _HASH_SCHEME, 1, _now()))
        return _public(conn.execute('SELECT * FROM auth_accounts WHERE username=?',
                                    (username,)).fetchone())


def list_accounts(db_path: str | Path, actor_token: str) -> list[dict]:
    """Only the owner may read the local account directory."""
    with _transaction(db_path) as conn:
        _require_owner(conn, actor_token)
        return [_public(row) for row in conn.execute('SELECT * FROM auth_accounts ORDER BY username')]


def change_password(db_path: str | Path, token: str, current_password: str, new_password: str) -> None:
    """Verify the current password, replace its hash, and revoke every own session."""
    with _transaction(db_path) as conn:
        row = _require_user(conn, token)
        try:
            encoded = _password(current_password)
        except ValueError:
            raise PermissionError('The current password could not be verified.') from None
        _password(new_password)
        if not _reserve_attempt(conn, row['username'], _now()):
            reserved = False
        else:
            reserved = True
    if not reserved or not _matches(encoded, row):
        raise PermissionError('The current password could not be verified; try again later.')
    if hmac.compare_digest(current_password.encode('utf-8'), new_password.encode('utf-8')):
        raise ValueError('Choose a different new passphrase.')
    salt, password_hash = _credentials(new_password)
    with _transaction(db_path) as conn:
        fresh = _require_user(conn, token)
        if fresh['security_version'] != row['security_version']:
            raise PermissionError('Sign in again to continue.')
        conn.execute('''UPDATE auth_accounts SET salt=?,password_hash=?,hash_scheme=?,
            security_version=security_version+1 WHERE username=?''',
                     (salt, password_hash, _HASH_SCHEME, row['username']))
        conn.execute('DELETE FROM auth_sessions WHERE username=?', (row['username'],))
        conn.execute('DELETE FROM auth_throttles WHERE bucket_key=?', (_bucket(row['username']),))


def disable_account(db_path: str | Path, actor_token: str, username: str) -> None:
    """Disable a local account immediately; the last active owner is protected."""
    username = _username(username)
    with _transaction(db_path) as conn:
        _require_owner(conn, actor_token)
        row = conn.execute('SELECT * FROM auth_accounts WHERE username=?', (username,)).fetchone()
        if row is None:
            raise ValueError('Choose an existing local account.')
        if row['role'] == 'owner' and row['active']:
            count = conn.execute("SELECT COUNT(*) FROM auth_accounts WHERE role='owner' AND active=1"
                                 ).fetchone()[0]
            if count <= 1:
                raise ValueError('The last active workspace owner cannot be disabled.')
        conn.execute('''UPDATE auth_accounts SET active=0,security_version=security_version+1
                        WHERE username=?''', (username,))
        conn.execute('DELETE FROM auth_sessions WHERE username=?', (username,))


def reset_account_password(db_path: str | Path, actor_token: str, username: str,
                           new_password: str) -> None:
    """The owner may reset an active accountant's password, revoking their sessions.

    This is local administration, without email or password disclosure. The owner
    must use change_password with their current password for their own account.
    """
    username = _username(username)
    with _transaction(db_path) as conn:
        _require_owner(conn, actor_token)
        target = conn.execute('SELECT * FROM auth_accounts WHERE username=?', (username,)).fetchone()
        if target is None or target['role'] != 'accountant' or not target['active']:
            raise ValueError('Choose an active accountant account.')
    salt, password_hash = _credentials(new_password)
    with _transaction(db_path) as conn:
        _require_owner(conn, actor_token)
        fresh = conn.execute('SELECT * FROM auth_accounts WHERE username=?', (username,)).fetchone()
        if (not fresh or not fresh['active'] or fresh['role'] != 'accountant' or
                fresh['security_version'] != target['security_version']):
            raise PermissionError('The account changed; refresh the local account directory.')
        conn.execute('''UPDATE auth_accounts SET salt=?,password_hash=?,hash_scheme=?,
            security_version=security_version+1 WHERE username=?''',
                     (salt, password_hash, _HASH_SCHEME, username))
        conn.execute('DELETE FROM auth_sessions WHERE username=?', (username,))
        conn.execute('DELETE FROM auth_throttles WHERE bucket_key=?', (_bucket(username),))
