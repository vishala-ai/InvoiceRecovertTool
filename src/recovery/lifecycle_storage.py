"""Versioned local synthetic evidence storage and meaningful input identities."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from .storage import connect, metadata
from .schemas import TABLE_SCHEMAS

CALCULATION_VERSION = 'hourly-decimal-v1-lifecycle1'
SCOPE = ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def exists(conn, table):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def active(conn):
    return exists(conn, 'lifecycle_migrations')


def synthetic(conn):
    manifest = json.loads(metadata(conn, 'manifest'))
    if manifest.get('synthetic') is not True:
        raise ValueError('SYNTHETIC_ONLY: this feature accepts synthetic snapshots only')
    return manifest


def scope_of(row):
    return {key: row[key] for key in SCOPE}


def scope_id(scope):
    return digest(scope_of(scope))


def scope_fingerprint(conn, scope, *, include_requests=True):
    """Hash input facts/versions, excluding display times and model proposals."""
    where = ' AND '.join(f'{key}=?' for key in SCOPE)
    args = tuple(scope[key] for key in SCOPE)
    facts = {'calculation_version': CALCULATION_VERSION,
             'manifest': json.loads(metadata(conn, 'manifest')),
             'fixture_source_digest': metadata(conn, 'source_digest')}
    for table in ('documents', 'reviewed_terms', 'invoice_lines'):
        facts[table] = sorted([dict(row) for row in conn.execute(
            f'SELECT * FROM {table} WHERE {where}', args)], key=encode)
    for table in ('service_entries', 'payments', 'adjustments'):
        # Keep the same complete scope and canonical ordering without loading
        # every other supplier's transactions for each invoice assessment.
        scoped_where = ' AND '.join(f'l.{key}=?' for key in SCOPE)
        facts[table] = sorted([dict(row) for row in conn.execute(
            f'SELECT t.* FROM {table} t JOIN invoice_lines l '
            f'ON l.tenant_id=t.tenant_id AND l.line_id=t.line_id WHERE {scoped_where}', args)], key=encode)
    facts['purchase_orders'] = sorted([dict(row) for row in conn.execute('''SELECT * FROM
        purchase_orders WHERE tenant_id=? AND supplier_id=? AND contract_id=? AND role=?''',
        tuple(scope[key] for key in SCOPE if key != 'location'))], key=encode)
    if active(conn):
        facts['source_versions'] = [tuple(row) for row in conn.execute('''SELECT version_id,
            sha256 FROM source_versions WHERE scope_id=? ORDER BY version_id''', (scope_id(scope),))]
        facts['source_dispositions'] = [tuple(row) for row in conn.execute('''SELECT version_id,
            decision,reviewer,note FROM source_dispositions WHERE scope_id=? ORDER BY event_id''',
            (scope_id(scope),))]
        facts['term_versions'] = [tuple(row) for row in conn.execute('''SELECT version_id,
            values_json FROM approved_term_versions WHERE scope_id=? ORDER BY version_id''',
            (scope_id(scope),))]
        if include_requests:
            scoped_where = ' AND '.join(f'l.{key}=?' for key in SCOPE)
            facts['requests'] = [tuple(row) for row in conn.execute(f'''SELECT r.request_id,r.line_id,
                r.request_type,r.details,r.status FROM evidence_requests r JOIN invoice_lines l
                ON l.tenant_id=r.tenant_id AND l.line_id=r.line_id WHERE {scoped_where}
                ORDER BY r.request_id''', args)]
    return digest(facts)


def input_revision(conn, line):
    if not active(conn):
        return None
    state = conn.execute('SELECT baseline_digest FROM scope_state WHERE scope_id=?',
                         (scope_id(line),)).fetchone()
    current = scope_fingerprint(conn, line)
    return current if not state or current != state['baseline_digest'] else None


def touch_scope(conn, scope, reason):
    key = scope_id(scope)
    conn.execute('''INSERT INTO scope_events VALUES (?,?,?,?,?)''',
                 ('scope:' + uuid.uuid4().hex, key, reason, now(), scope['tenant_id']))


def open_requests(conn, tenant, line_id):
    return active(conn) and conn.execute('''SELECT 1 FROM evidence_requests WHERE tenant_id=?
        AND line_id=? AND status!='RESOLVED' LIMIT 1''', (tenant, line_id)).fetchone() is not None


def coverage(conn, docs):
    """Return reviewed/disposed additive doc IDs; reject incomplete coverage."""
    reviewed, ignored = set(), set()
    if not active(conn):
        return reviewed, ignored
    for doc in docs:
        source = conn.execute('''SELECT * FROM source_versions WHERE tenant_id=? AND doc_id=?''',
                              (doc['tenant_id'], doc['doc_id'])).fetchone()
        if not source or source['origin'] == 'CURATED_FIXTURE':
            continue
        if hashlib.sha256(doc['body'].encode()).hexdigest() != source['sha256']:
            raise ValueError('SOURCE_VERSION_MISMATCH')
        decision = conn.execute('''SELECT decision FROM source_dispositions WHERE version_id=?
            ORDER BY sequence DESC LIMIT 1''', (source['version_id'],)).fetchone()
        if decision and decision['decision'] in ('NO_PRICING_EFFECT', 'OUT_OF_SCOPE'):
            ignored.add(doc['doc_id'])
        elif conn.execute('''SELECT 1 FROM term_review_events WHERE source_version=?
                AND decision='TERMS_APPROVED' AND coverage_confirmed=1''',
                          (source['version_id'],)).fetchone():
            reviewed.add(doc['doc_id'])
    return reviewed, ignored


def validate_source_projections(conn):
    """Check source classifications against existing import/version provenance.

    Original source versions predate a separate ``kind`` field. The retained
    import reconciliation digest nevertheless covers every original document
    field, including kind, path and body. Compare only curated fixture versions
    with that complete baseline, so legitimate intake additions remain separate.
    Intake admission itself fixes its document kind to RATE_AGREEMENT.

    This is a read-only integrity check, not authentication of editable SQLite
    metadata or independent verification of synthetic source content.
    """
    if not active(conn):
        return
    try:
        report = json.loads(metadata(conn, 'import_report'))
        baseline = report['reconciliation']['source']
        count = baseline['row_counts']['documents']
        expected_digest = baseline['row_digests']['documents']
        if (type(count) is not int or count < 0
                or not isinstance(expected_digest, str) or len(expected_digest) != 64
                or any(character not in '0123456789abcdef' for character in expected_digest)):
            raise ValueError('Invalid retained source reconciliation')
        documents = [dict(row) for row in conn.execute('''SELECT d.* FROM source_versions s
            JOIN documents d ON d.tenant_id=s.tenant_id AND d.doc_id=s.doc_id
            WHERE s.origin='CURATED_FIXTURE' ''')]
        fields = [*TABLE_SCHEMAS['documents'], 'body']
        records = sorted([[document[field] for field in fields] for document in documents])
        # Match storage._metrics exactly; its original digest uses these default
        # JSON separators and preserves Unicode rather than normalizing source.
        actual_digest = hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest()
        if len(documents) != count or actual_digest != expected_digest:
            raise ValueError('Original document projection changed')
        if conn.execute('''SELECT 1 FROM source_versions s LEFT JOIN documents d
            ON d.tenant_id=s.tenant_id AND d.doc_id=s.doc_id
            WHERE s.origin='SYNTHETIC_INTAKE' AND
                (d.kind IS NULL OR d.kind!='RATE_AGREEMENT') LIMIT 1''').fetchone():
            raise ValueError('Intake document classification changed')
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ValueError('SOURCE_PROJECTION_MISMATCH') from exc


def validate_projections(conn, line):
    """Known evidence cannot disappear merely by changing its searchable projection."""
    if not active(conn):
        return
    validate_source_projections(conn)
    for source in conn.execute('SELECT * FROM source_versions WHERE scope_id=?', (scope_id(line),)):
        doc=conn.execute('SELECT * FROM documents WHERE tenant_id=? AND doc_id=?',
                         (source['tenant_id'],source['doc_id'])).fetchone()
        if (not doc or scope_of(doc)!=json.loads(source['scope_json'])
                or doc['sha256']!=source['sha256'] or doc['body']!=source['body']
                or doc['status']!=source['status'] or doc['signed_date']!=source['signed_date']
                or hashlib.sha256(source['body'].encode()).hexdigest()!=source['sha256']):
            raise ValueError('SOURCE_PROJECTION_MISMATCH')
    for version in conn.execute('SELECT * FROM approved_term_versions WHERE scope_id=?',(scope_id(line),)):
        term=conn.execute('SELECT * FROM reviewed_terms WHERE tenant_id=? AND term_id=?',
                          (version['tenant_id'],version['term_id'])).fetchone()
        if not term or dict(term)!=json.loads(version['values_json']):
            raise ValueError('TERM_PROJECTION_MISMATCH')
    where=' AND '.join(key+'=?' for key in SCOPE)
    for term in conn.execute('SELECT * FROM reviewed_terms WHERE '+where+
                             " AND reviewed='true' AND reviewer!=''", tuple(line[key] for key in SCOPE)):
        version=conn.execute('SELECT values_json FROM approved_term_versions WHERE tenant_id=? AND term_id=?',
                             (term['tenant_id'],term['term_id'])).fetchone()
        if not version or dict(term)!=json.loads(version[0]):
            raise ValueError('TERM_PROJECTION_MISMATCH')


@contextmanager
def transaction(db):
    with connect(db) as conn, conn:
        if not active(conn):
            raise ValueError('Run recover migrate before using the evidence lifecycle')
        synthetic(conn)
        conn.execute('BEGIN IMMEDIATE')
        yield conn


@contextmanager
def review_start_lock(db, thread_id):
    """A process-crash-safe local lock; never hold a database write lock over inference."""
    import fcntl
    path = Path(db).with_name(Path(db).stem + '-start-' + digest(thread_id)[:24] + '.lock')
    if path.is_symlink():
        raise ValueError('Review start lock must not be a symlink')
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def backup_database(db, directory):
    raw_source = Path(db)
    if raw_source.is_symlink():
        raise ValueError('Backup source must not be a symlink')
    source = raw_source.resolve()
    if not source.is_file():
        raise ValueError('Backup requires an existing regular database')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (source.stem + '-' + uuid.uuid4().hex + '.sqlite3')
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(target) as backup:
            original.backup(backup)
            if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Backup integrity check failed')
            if original.execute('SELECT count(*) FROM review_events').fetchone()[0] != backup.execute(
                    'SELECT count(*) FROM review_events').fetchone()[0]:
                raise ValueError('Backup journal verification failed')
    return str(target)


DDL = [
    '''CREATE TABLE lifecycle_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)''',
    '''CREATE TABLE scope_state(scope_id TEXT PRIMARY KEY,scope_json TEXT NOT NULL,
        baseline_digest TEXT NOT NULL)''',
    '''CREATE TABLE scope_events(event_id TEXT PRIMARY KEY,scope_id TEXT NOT NULL,
        reason TEXT NOT NULL,recorded_at TEXT NOT NULL,tenant_id TEXT NOT NULL)''',
    '''CREATE TABLE source_stages(stage_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        original_filename TEXT NOT NULL,body TEXT NOT NULL,sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,UNIQUE(tenant_id,sha256))''',
    '''CREATE TABLE source_versions(version_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,family_id TEXT NOT NULL,scope_id TEXT NOT NULL,scope_json TEXT NOT NULL,
        sha256 TEXT NOT NULL,body TEXT NOT NULL,original_filename TEXT NOT NULL,
        status TEXT NOT NULL,signed_date TEXT NOT NULL,title TEXT NOT NULL,origin TEXT NOT NULL,
        previous_version TEXT,created_at TEXT NOT NULL,UNIQUE(tenant_id,doc_id))''',
    '''CREATE TABLE operation_keys(tenant_id TEXT NOT NULL,operation_key TEXT NOT NULL,
        payload_digest TEXT NOT NULL,result_json TEXT NOT NULL,PRIMARY KEY(tenant_id,operation_key))''',
    '''CREATE TABLE term_proposals(proposal_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,source_version TEXT NOT NULL,scope_json TEXT NOT NULL,
        context_digest TEXT NOT NULL,revision INTEGER NOT NULL,parent_id TEXT,
        values_json TEXT NOT NULL,original_json TEXT NOT NULL,metadata_json TEXT NOT NULL,
        reviewer_note TEXT NOT NULL,digest TEXT NOT NULL,created_at TEXT NOT NULL)''',
    '''CREATE TABLE term_review_events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,tenant_id TEXT NOT NULL,proposal_id TEXT NOT NULL,
        source_version TEXT NOT NULL,reviewed_digest TEXT NOT NULL,decision TEXT NOT NULL,
        reviewer TEXT NOT NULL,note TEXT NOT NULL,term_version TEXT,coverage_confirmed INTEGER NOT NULL,
        recorded_at TEXT NOT NULL)''',
    '''CREATE TABLE approved_term_versions(version_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        term_id TEXT NOT NULL,scope_id TEXT NOT NULL,values_json TEXT NOT NULL,
        source_version TEXT NOT NULL,proposal_id TEXT,review_event_id TEXT,origin TEXT NOT NULL,
        created_at TEXT NOT NULL,UNIQUE(tenant_id,term_id))''',
    '''CREATE TABLE source_dispositions(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,tenant_id TEXT NOT NULL,version_id TEXT NOT NULL,
        scope_id TEXT NOT NULL,decision TEXT NOT NULL,reviewer TEXT NOT NULL,note TEXT NOT NULL,
        recorded_at TEXT NOT NULL)''',
    '''CREATE TABLE assessment_revisions(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        assessment_id TEXT UNIQUE NOT NULL,tenant_id TEXT NOT NULL,line_id TEXT NOT NULL,
        input_digest TEXT NOT NULL,case_digest TEXT NOT NULL,case_json TEXT NOT NULL,
        prior_assessment_id TEXT,prior_review_event_id TEXT,created_at TEXT NOT NULL)''',
    '''CREATE TABLE evidence_requests(request_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        line_id TEXT NOT NULL,review_event_id TEXT NOT NULL,assessment_id TEXT,
        request_type TEXT NOT NULL,details TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,
        UNIQUE(tenant_id,review_event_id))''',
    '''CREATE TABLE evidence_request_events(event_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        request_id TEXT NOT NULL,version_id TEXT,decision TEXT NOT NULL,reviewer TEXT NOT NULL,
        note TEXT NOT NULL,recorded_at TEXT NOT NULL)''',
    '''CREATE TABLE case_review_assessments(event_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        assessment_id TEXT NOT NULL,thread_id TEXT NOT NULL)''',
    '''CREATE TABLE review_intents(intent_id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,
        line_id TEXT NOT NULL,assessment_id TEXT NOT NULL,thread_id TEXT NOT NULL UNIQUE,
        prior_thread_id TEXT,config_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL)''',
]


def migrate(db, *, backup_dir=None):
    """Add schema only after a verified SQLite backup; never replace source tables."""
    with connect(db) as conn:
        synthetic(conn)
        if active(conn):
            version = conn.execute('SELECT max(version) FROM lifecycle_migrations').fetchone()[0]
            if version != 1:
                raise ValueError('Unsupported lifecycle schema version')
            return {'version': version, 'already_migrated': True, 'backup': None}
    backup = backup_database(db, backup_dir or Path(db).parent / 'backups')
    with connect(db) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if active(conn):
            return {'version': 1, 'already_migrated': True, 'backup': backup}
        for statement in DDL:
            conn.execute(statement)
        for table in ('source_versions','term_proposals','term_review_events','approved_term_versions',
                      'assessment_revisions','evidence_request_events','source_dispositions',
                      'review_events','case_review_assessments'):
            for operation in ('UPDATE','DELETE'):
                conn.execute(f'''CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation}
                    ON {table} BEGIN SELECT RAISE(ABORT,'Historical records are append-only'); END''')
        for doc in conn.execute('SELECT * FROM documents').fetchall():
            scope = scope_of(doc)
            version = 'fixture-doc:' + digest([doc['tenant_id'], doc['doc_id'], doc['sha256']])
            conn.execute('INSERT INTO source_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (version,doc['tenant_id'],doc['doc_id'],doc['doc_id'],scope_id(scope),encode(scope),
                 doc['sha256'],doc['body'],doc['path'],doc['status'],doc['signed_date'],
                 doc['doc_id'],'CURATED_FIXTURE',None,now()))
        for term in conn.execute('SELECT * FROM reviewed_terms').fetchall():
            if term['reviewed'] != 'true' or not term['reviewer']:
                continue
            source = conn.execute('SELECT version_id FROM source_versions WHERE tenant_id=? AND doc_id=?',
                                  (term['tenant_id'],term['doc_id'])).fetchone()
            conn.execute('INSERT INTO approved_term_versions VALUES (?,?,?,?,?,?,?,?,?,?)',
                ('fixture-term:' + digest(dict(term)),term['tenant_id'],term['term_id'],
                 scope_id(term),encode(dict(term)),source[0],None,None,'CURATED_FIXTURE',now()))
        scopes = {scope_id(row):scope_of(row) for row in conn.execute('SELECT * FROM invoice_lines')}
        for key,scope in scopes.items():
            conn.execute('INSERT INTO scope_state VALUES (?,?,?)',
                         (key,encode(scope),scope_fingerprint(conn,scope)))
        conn.execute('INSERT INTO lifecycle_migrations VALUES (1,?)', (now(),))
        from .engine import evaluate_line
        for row in conn.execute('SELECT tenant_id,line_id FROM invoice_lines').fetchall():
            case = evaluate_line(conn,row['tenant_id'],row['line_id'])
            save_assessment(conn,case)
    return {'version': 1, 'already_migrated': False, 'backup': backup}


def save_assessment(conn, case, *, prior_review_event_id=None):
    from .legacy_archive import assert_not_archived
    assert_not_archived(conn, case['tenant_id'], case['line_id'])
    previous = conn.execute('''SELECT assessment_id FROM assessment_revisions WHERE tenant_id=?
        AND line_id=? ORDER BY sequence DESC LIMIT 1''', (case['tenant_id'],case['line_id'])).fetchone()
    record = {'assessment_id':'assessment:' + uuid.uuid4().hex,'tenant_id':case['tenant_id'],
              'line_id':case['line_id'],'input_digest':case.get('input_revision',case['case_digest']),
              'case_digest':case['case_digest'],'case_json':case,
              'prior_assessment_id':previous[0] if previous else None,
              'prior_review_event_id':prior_review_event_id,'created_at':now()}
    conn.execute('''INSERT INTO assessment_revisions(assessment_id,tenant_id,line_id,input_digest,
        case_digest,case_json,prior_assessment_id,prior_review_event_id,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)''', tuple(encode(value) if key=='case_json' else value
                                           for key,value in record.items()))
    return record
