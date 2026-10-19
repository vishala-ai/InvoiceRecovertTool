"""Prepare a source-only synthetic repository without publishing or reading secrets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'var/cloud-implementation/InvoiceRecovertTool'
ROOT_FILES = ('app.py', 'pyproject.toml', 'uv.lock', 'README.md', 'AGENTS.md', '.gitignore')
SOURCE_DIRECTORIES = ('src', 'data', 'docs', 'tests', 'scripts', 'cloud')
EXTENSIONS = {'.py', '.md', '.csv', '.json', '.jsonl', '.toml', '.txt', '.sql'}
EXCLUDED_DIRECTORIES = {'var', 'examples', '__pycache__', 'node_modules', 'backups'}
CONFIG = '.streamlit/config.toml'
EXAMPLE = 'cloud/secrets.example.toml'
MANIFEST = 'PREPARED_FILES.json'
MAX_FILE_BYTES = 8 * 1024 * 1024
SAFE_EXAMPLE = {
    'cloud': {
        'workspace_id': 'invoice-recovery-demo',
        'database_url': 'postgresql://APP_ROLE:PRIVATE_PASSWORD@YOUR_SESSION_POOLER_HOST:5432/postgres',
        'root_cert_pem': '-----BEGIN CERTIFICATE-----\n'
                         'PASTE_THE_PUBLIC_SUPABASE_SERVER_ROOT_CERTIFICATE\n'
                         '-----END CERTIFICATE-----',
        'owner': {'username': 'choose-an-owner', 'display_name': 'Cloud workspace owner',
                  'password': 'REPLACE_WITH_A_UNIQUE_PRIVATE_PASSPHRASE'},
    },
    'groq': {'api_key': 'SET_PRIVATELY_IN_STREAMLIT_SECRETS', 'model': 'llama-3.3-70b-versatile'},
}


def _excluded_name(name):
    lowered = name.casefold()
    return (name.startswith('.') or '.env' in lowered
            or lowered in EXCLUDED_DIRECTORIES
            or bool(re.search(r'secret|credential|password|private.?key|access.?token|api.?key', lowered))
            or lowered in {'accounts.json', 'accounts.csv', 'users.json', 'users.csv'})


def _read_regular(path):
    """Read only an admitted regular non-link file, including during path races."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_FILE_BYTES:
            raise ValueError('Repository inputs must be bounded regular files without aliases')
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            content = source.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError('Repository input exceeds the source file size limit')
        return content
    finally:
        os.close(descriptor)


def _source_files(source):
    for name in ROOT_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Required source file is missing or linked: {name}')
        yield Path(name)

    def walk(directory):
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            relative = child.relative_to(source)
            if child.is_symlink():
                continue
            if relative.as_posix() == EXAMPLE:
                if child.is_file():
                    yield relative
                continue
            if _excluded_name(child.name):
                continue
            if child.is_dir():
                yield from walk(child)
            elif child.is_file() and child.suffix.casefold() in EXTENSIONS:
                yield relative

    for name in SOURCE_DIRECTORIES:
        directory = source / name
        if directory.is_symlink():
            continue
        if directory.is_dir():
            yield from walk(directory)
    config = source / CONFIG
    if (source / '.streamlit').is_symlink() or config.is_symlink() or not config.is_file():
        raise ValueError('A regular local Streamlit configuration is required')
    yield Path(CONFIG)


def _cloud_configuration(content):
    text = content.decode('utf-8')
    parsed = tomllib.loads(text)
    # Preserve comments, theme and telemetry text verbatim; omit the local bind
    # settings only in the published copy so Cloud selects its own address/port.
    result = []
    in_server = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith('['):
            match = re.fullmatch(r'\[\s*([^\]]+)\s*\]\s*(?:#.*)?(?:\r?\n)?', stripped)
            if not match:
                raise ValueError('Unsupported Streamlit configuration table syntax')
            section = match[1].strip()
            in_server = section == 'server' or section.startswith('server.')
        if not in_server:
            result.append(line)
    cloud_text = ''.join(result)
    expected = {key: value for key, value in parsed.items() if key != 'server'}
    if tomllib.loads(cloud_text) != expected:
        raise ValueError('Cloud configuration did not preserve the other local settings')
    return cloud_text.encode('utf-8')


def _validate_manifest(content):
    # The same explicit input schema as the importer is authoritative. This
    # command supports direct Python invocation without loading any .env file.
    try:
        from recovery.schemas import validate_manifest
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / 'src'))
        from recovery.schemas import validate_manifest

    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Synthetic manifest fields must be unique')
            value[key] = item
        return value

    manifest = json.loads(content.decode('utf-8'), object_pairs_hook=unique_fields)
    if validate_manifest(manifest):
        raise ValueError('Repository preparation requires a valid synthetic-only fixture manifest')
    return manifest


def prepare_repository(source=ROOT, output=DEFAULT_OUTPUT):
    source, output = Path(source).absolute(), Path(output).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Source must be an existing regular project directory')
    if output.exists() or output.is_symlink():
        raise ValueError('Output already exists; choose a new repository staging directory')
    if output == source or source.is_relative_to(output):
        raise ValueError('Output must not replace the source project or its parent')
    manifest_path = source / 'data/manifest.json'
    if (source / 'data').is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('A regular synthetic fixture manifest is required')
    fixture_content = _read_regular(manifest_path)
    fixture_manifest = _validate_manifest(fixture_content)
    files = list(_source_files(source))
    required = {'data/manifest.json', 'cloud/streamlit_app.py', 'cloud/requirements.txt', EXAMPLE}
    if not required.issubset({path.as_posix() for path in files}):
        raise ValueError('Cloud entrypoint, resolved requirements and placeholder secrets example are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cloud-repository-', dir=output.parent))
    reserved = False
    try:
        entries = []
        for relative in sorted(files, key=lambda path: path.as_posix()):
            content = _read_regular(source / relative)
            original_hash = hashlib.sha256(content).hexdigest()
            name = relative.as_posix()
            if name == 'data/manifest.json' and content != fixture_content:
                raise ValueError('Synthetic fixture manifest changed during repository preparation')
            if name == EXAMPLE and tomllib.loads(content.decode('utf-8')) != SAFE_EXAMPLE:
                raise ValueError('The public secrets example must contain only the approved placeholders')
            if name == CONFIG:
                content = _cloud_configuration(content)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            if hashlib.sha256(_read_regular(target)).hexdigest() != digest:
                raise ValueError('Prepared source verification failed')
            entries.append({'path': name, 'sha256': digest, 'source_sha256': original_hash,
                            'bytes': len(content), 'transformed': name == CONFIG})
        for entry in entries:
            if (hashlib.sha256(_read_regular(source / entry['path'])).hexdigest()
                    != entry['source_sha256']
                    or hashlib.sha256(_read_regular(staging / entry['path'])).hexdigest()
                    != entry['sha256']):
                raise ValueError('Repository sources changed before publication')
        result = {'schema_version': 1, 'synthetic': True,
                  'dataset_id': fixture_manifest['dataset_id'],
                  'file_count': len(entries), 'manifest_file': MANIFEST,
                  'files': entries}
        (staging / MANIFEST).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        # Exclusive reservation prevents replacing even a concurrently created
        # empty destination. Only this new empty directory is atomically swapped.
        output.mkdir()
        reserved = True
        os.replace(staging, output)
        reserved = False
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if reserved:
            try:
                output.rmdir()
            except OSError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = prepare_repository(args.source, args.output)
    except (OSError, ValueError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        parser.exit(2, f'Repository not prepared: {exc}\n')
    print(json.dumps({'status': 'PREPARED', 'synthetic': True,
                      'dataset_id': report['dataset_id'], 'file_count': report['file_count'],
                      'published': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
