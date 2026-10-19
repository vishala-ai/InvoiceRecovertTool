"""Human-controlled synthetic term review, evidence intake and reassessment.

Provider calls occur outside write transactions. Every promotion or disposition
checks authoritative scope and freshness again inside its transaction.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from .storage import connect, one_line
from .lifecycle_storage import (SCOPE, active, coverage, digest, encode, migrate, now,
    save_assessment, scope_fingerprint, scope_id, scope_of, synthetic, touch_scope, transaction)

__all__ = ['migrate']
TERM_FIELDS = ('role','location','currency','unit','service_type','rate','effective_from',
               'effective_to','supersedes','exact_quote','unresolved_questions','coverage')


def enabled(db):
    if not Path(db).is_file():
        return False
    with connect(db) as conn:
        return active(conn)


def _require_text(value, label):
    if not isinstance(value,str) or not value.strip() or len(value)>6000:
        raise ValueError(f'{label} is required and must be text')
    return value.strip()


def _reviewer(reviewer,note):
    _require_text(reviewer,'Demo reviewer label')
    _require_text(note,'Review note')


def _scope(conn, tenant, scope):
    synthetic(conn)
    required=set(SCOPE)-{'tenant_id'}
    if (not isinstance(scope,dict) or not required.issubset(scope)
            or set(scope)-required-{'tenant_id'}
            or any(not isinstance(scope[key],str) or not scope[key].strip() for key in required)):
        raise ValueError('Scope requires exactly supplier, contract, role and location IDs')
    result = {'tenant_id':tenant, **{key:scope[key] for key in SCOPE if key!='tenant_id'}}
    if scope.get('tenant_id',tenant)!=tenant:
        raise ValueError('Evidence tenant does not match the selected tenant')
    where=' AND '.join(key+'=?' for key in SCOPE)
    if not conn.execute('SELECT 1 FROM invoice_lines WHERE '+where,
                        tuple(result[key] for key in SCOPE)).fetchone():
        raise ValueError('Scope must exactly match an existing synthetic invoice scope')
    return result


def _previous(conn,tenant,key,payload):
    _require_text(key,'Idempotency key')
    row=conn.execute('SELECT * FROM operation_keys WHERE tenant_id=? AND operation_key=?',
                     (tenant,key)).fetchone()
    if not row:
        return None
    if row['payload_digest']!=digest(payload):
        raise ValueError('Idempotency key already used for a different payload')
    return json.loads(row['result_json'])


def _remember(conn,tenant,key,payload,result):
    conn.execute('INSERT INTO operation_keys VALUES (?,?,?,?)',
                 (tenant,key,digest(payload),encode(result)))
    return result


def _source(conn,tenant,version):
    row=conn.execute('SELECT * FROM source_versions WHERE tenant_id=? AND version_id=?',
                     (tenant,version)).fetchone()
    if not row:
        raise ValueError('Source version not found in the selected tenant')
    result=dict(row)
    result['scope']=json.loads(result.pop('scope_json'))
    current=conn.execute('SELECT * FROM documents WHERE tenant_id=? AND doc_id=?',
                         (tenant,result['doc_id'])).fetchone()
    if (not current or current['sha256']!=result['sha256'] or current['body']!=result['body']
            or scope_of(current)!=result['scope'] or current['status']!=result['status']
            or current['signed_date']!=result['signed_date']
            or hashlib.sha256(result['body'].encode()).hexdigest()!=result['sha256']):
        raise ValueError('Source version or projection changed; fresh review required')
    result['disposition']=None
    disposition=conn.execute('''SELECT decision FROM source_dispositions WHERE tenant_id=?
        AND version_id=? ORDER BY sequence DESC LIMIT 1''',(tenant,version)).fetchone()
    if disposition:
        result['disposition']=disposition[0]
    return result


def list_sources(db,tenant):
    with connect(db) as conn:
        return [_source(conn,tenant,row[0]) for row in conn.execute('''SELECT version_id FROM
            source_versions WHERE tenant_id=? ORDER BY created_at DESC,version_id''',(tenant,))]


def list_documents(db,tenant):
    with connect(db) as conn:
        rows=[]
        for row in conn.execute('SELECT * FROM documents WHERE tenant_id=? ORDER BY doc_id',(tenant,)):
            doc=dict(row)
            source=conn.execute('SELECT version_id FROM source_versions WHERE tenant_id=? AND doc_id=?',
                                (tenant,doc['doc_id'])).fetchone()
            doc['version_id']=doc['source_version']=source[0] if source else None
            rows.append(doc)
        return rows


def _proposal(conn,tenant,proposal_id):
    row=conn.execute('SELECT * FROM term_proposals WHERE tenant_id=? AND proposal_id=?',
                     (tenant,proposal_id)).fetchone()
    if not row:
        raise ValueError('Proposal not found in the selected tenant')
    result=dict(row)
    for source,target in [('scope_json','scope'),('values_json','values'),
                           ('original_json','original_output'),('metadata_json','metadata')]:
        result[target]=json.loads(result.pop(source))
    decision=conn.execute('''SELECT * FROM term_review_events WHERE tenant_id=? AND proposal_id=?
        ORDER BY sequence DESC LIMIT 1''',(tenant,proposal_id)).fetchone()
    result['status']=decision['decision'] if decision else 'UNREVIEWED_PROPOSAL'
    try:
        _source(conn,tenant,result['source_version'])
        current=scope_fingerprint(conn,result['scope'],include_requests=False)
    except ValueError:
        current=None
    if not decision and current!=result['context_digest']:
        result['status']='PROPOSAL_STALE'
    result['review']=dict(decision) if decision else None
    source_hash=conn.execute('SELECT sha256 FROM source_versions WHERE tenant_id=? AND version_id=?',
                            (tenant,result['source_version'])).fetchone()
    result['source_sha256']=source_hash[0] if source_hash else None
    return result


def list_proposals(db,tenant):
    with connect(db) as conn:
        return [_proposal(conn,tenant,row[0]) for row in conn.execute('''SELECT proposal_id FROM
            term_proposals WHERE tenant_id=? ORDER BY created_at DESC,revision DESC''',(tenant,))]


def _insert_proposal(conn,tenant,document,values,original,metadata,context,*,parent=None,note=''):
    from .term_extraction import validate_proposal
    values=validate_proposal(values)
    scope=scope_of(document)
    record={'proposal_id':'proposal:'+uuid.uuid4().hex,'tenant_id':tenant,'doc_id':document['doc_id'],
            'source_version':document['version_id'],'scope_json':encode(scope),
            'context_digest':context,'revision':parent['revision']+1 if parent else 1,
            'parent_id':parent['proposal_id'] if parent else None,'values_json':encode(values),
            'original_json':encode(original),'metadata_json':encode(metadata),'reviewer_note':note}
    record['digest']=digest(record)
    record['created_at']=now()
    conn.execute('INSERT INTO term_proposals VALUES ('+','.join('?' for _ in record)+')',
                 tuple(record.values()))
    return _proposal(conn,tenant,record['proposal_id'])


def generate_proposal(db,tenant,doc_id,*,model='llama3.2:latest',
                      base_url='http://127.0.0.1:11434',context_tokens=16384):
    from .term_extraction import extract_terms
    with connect(db) as conn:
        if not active(conn):
            raise ValueError('Run recover migrate first')
        synthetic(conn)
        row=conn.execute('''SELECT d.*,s.version_id FROM documents d JOIN source_versions s
            ON d.tenant_id=s.tenant_id AND d.doc_id=s.doc_id WHERE d.tenant_id=? AND d.doc_id=?''',
            (tenant,doc_id)).fetchone()
        if not row:
            raise ValueError('Document not found in the selected tenant')
        document=dict(row)
        _source(conn,tenant,document['version_id'])
        context=scope_fingerprint(conn,document,include_requests=False)
    output=extract_terms(document,model=model,base_url=base_url,context_tokens=context_tokens)
    if output.get('source_sha256')!=document['sha256']:
        raise ValueError('Extraction source changed; proposal withheld')
    metadata={key:value for key,value in output.items() if key not in ('proposal','original_output')}
    with transaction(db) as conn:
        # A late result is retained as stale, never silently rebound to new evidence.
        return _insert_proposal(conn,tenant,document,output['proposal'],
            output.get('original_output',output['proposal']),metadata,context)


def edit_proposal(db,tenant,proposal_id,values,reviewer,note,idempotency_key):
    _reviewer(reviewer,note)
    payload=['edit',proposal_id,values,reviewer,note]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        parent=_proposal(conn,tenant,proposal_id)
        if parent['status']!='UNREVIEWED_PROPOSAL':
            raise ValueError('Only a current unreviewed proposal can be edited; generate a replacement')
        document=_source(conn,tenant,parent['source_version'])
        result=_insert_proposal(conn,tenant,{**document,**document['scope']},values,
            parent['original_output'],{**parent['metadata'],'edited_by':reviewer},
            parent['context_digest'],parent=parent,note=note)
        return _remember(conn,tenant,idempotency_key,payload,result)


def _date(value,label,optional=False):
    if optional and value in (None,''):
        return ''
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        raise ValueError(f'Invalid {label}')
    date.fromisoformat(value)
    return value


def _approved_values(proposal,source,reviewer):
    values=proposal['values']
    if set(values)!=set(TERM_FIELDS):
        raise ValueError('Incomplete proposal fields')
    if values['unresolved_questions'] or values['coverage']!='COMPLETE_SINGLE_RATE':
        raise ValueError('Unresolved applicability or incomplete document coverage blocks approval')
    if any(values[key]!=source['scope'][key] for key in ('role','location')):
        raise ValueError('Proposed role/location does not match the reviewed source scope')
    if values['currency']!='USD' or values['unit']!='HOUR' or values['service_type']!='REGULAR':
        raise ValueError('Only USD regular hourly terms are supported')
    if not isinstance(values['rate'],str) or len(values['rate'])>80:
        raise ValueError('Rate must be a finite positive Decimal string')
    try:
        rate=Decimal(values['rate'])
    except InvalidOperation:
        raise ValueError('Rate must be a finite positive Decimal string') from None
    if not rate.is_finite() or rate<=Decimal('0') or rate>Decimal('1000000000') or rate.as_tuple().exponent < -8:
        raise ValueError('Rate must be finite, positive and within supported precision/bounds')
    quote=values['exact_quote']
    if not isinstance(quote,str) or not quote or quote not in source['body']:
        raise ValueError('An exact supporting source quotation is required')
    quoted_numbers=re.findall(r'(?<![\w.])\d+(?:\.\d+)?(?![\w.])',quote)
    if not any(Decimal(number)==rate for number in quoted_numbers):
        raise ValueError('The proposed rate must occur in the supporting quotation')
    start=_date(values['effective_from'],'effective start')
    end=_date(values['effective_to'],'effective end',True)
    if end and end<start:
        raise ValueError('Effective end precedes start')
    if start not in source['body'] or (end and end not in source['body']):
        raise ValueError('Effective dates must be supported by the source text')
    if 'USD' not in quote or 'HOUR' not in quote.upper():
        raise ValueError('Currency and hourly unit must be supported by the quoted rate clause')
    if values['supersedes'] and (not isinstance(values['supersedes'],str)
        or not re.search(r'(?<![\w:./-])'+re.escape(values['supersedes'])+r'(?![\w:/-]|\.\w)',source['body'])):
        raise ValueError('Explicit supersession target must be supported by the source text')
    return {**source['scope'],'currency':values['currency'],'unit':'HOUR','service_type':'REGULAR',
            'effective_from':start,'effective_to':end,'rate':values['rate'],
            'supersedes':values['supersedes'] or '', 'doc_id':source['doc_id'],
            'source_quote':quote,'reviewer':reviewer,'reviewed':'true'}


def review_terms(db,tenant,proposal_id,proposal_digest,decision,reviewer,note,idempotency_key,
                 coverage_confirmed=False):
    _reviewer(reviewer,note)
    if decision not in ('approve','reject'):
        raise ValueError('Choose approve or reject')
    payload=['terms',proposal_id,proposal_digest,decision,reviewer,note,coverage_confirmed]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        proposal=_proposal(conn,tenant,proposal_id)
        if proposal['digest']!=proposal_digest or proposal['status']!='UNREVIEWED_PROPOSAL':
            raise ValueError('Proposal is stale or already reviewed; refresh before deciding')
        if conn.execute('SELECT 1 FROM term_proposals WHERE parent_id=?',(proposal_id,)).fetchone():
            raise ValueError('A newer proposal revision exists; review that revision')
        source=_source(conn,tenant,proposal['source_version'])
        term_version=None
        event_id='term-review:'+uuid.uuid4().hex
        if decision=='approve':
            if coverage_confirmed is not True:
                raise ValueError('Explicit full single-rate document coverage confirmation is required')
            if source['status']!='SIGNED' or not source['signed_date']:
                raise ValueError('Unsigned documents cannot supply controlling approved terms')
            if source['disposition']:
                raise ValueError('A disposed source needs a new reviewed version before promotion')
            values=_approved_values(proposal,source,reviewer)
            existing=[dict(row) for row in conn.execute('''SELECT * FROM reviewed_terms
                WHERE tenant_id=? AND supplier_id=? AND contract_id=? AND role=? AND location=?''',
                tuple(values[key] for key in SCOPE))]
            comparable=[key for key in values if key not in ('reviewer','reviewed')]
            identical=next((term for term in existing if term['reviewed']=='true' and term['reviewer']
                and all((Decimal(term[key])==Decimal(values[key]) if key=='rate' else term[key]==values[key])
                        for key in comparable)),None)
            if identical:
                term_version=conn.execute('''SELECT version_id FROM approved_term_versions
                    WHERE tenant_id=? AND term_id=?''',(tenant,identical['term_id'])).fetchone()[0]
            else:
                supersedes=values['supersedes']
                if supersedes and not any(term['term_id']==supersedes for term in existing):
                    raise ValueError('Superseded term must exist in the exact tenant/rate scope')
                term={'term_id':'TERM-'+uuid.uuid4().hex,**values}
                from .engine import active_term
                eligible=[row for row in existing if row['reviewed']=='true' and row['reviewer']]
                # Validate all rate-period starts, including future terms, not upload order.
                for day in {item['effective_from'] for item in [*eligible,term]}:
                    active_term([*eligible,term],day,'USD')
                names=tuple(term)
                conn.execute('INSERT INTO reviewed_terms ('+','.join(names)+') VALUES ('+
                             ','.join('?' for _ in names)+')',tuple(term.values()))
                term_version='term-version:'+uuid.uuid4().hex
                conn.execute('INSERT INTO approved_term_versions VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (term_version,tenant,term['term_id'],scope_id(values),encode(term),
                     source['version_id'],proposal_id,event_id,'HUMAN_REVIEW',now()))
                touch_scope(conn,source['scope'],'TERMS_APPROVED')
        result={'event_id':event_id,'tenant_id':tenant,'proposal_id':proposal_id,
                'source_version':source['version_id'],'reviewed_digest':proposal_digest,
                'decision':'TERMS_APPROVED' if decision=='approve' else 'TERMS_REJECTED',
                'reviewer':reviewer,'note':note,'term_version':term_version,
                'coverage_confirmed':int(coverage_confirmed is True),'recorded_at':now()}
        conn.execute('''INSERT INTO term_review_events(event_id,tenant_id,proposal_id,source_version,
            reviewed_digest,decision,reviewer,note,term_version,coverage_confirmed,recorded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''',tuple(result.values()))
        return _remember(conn,tenant,idempotency_key,payload,result)


def stage_document(db,tenant,filename,content,synthetic_confirmed):
    if synthetic_confirmed is not True:
        raise ValueError('SYNTHETIC_ONLY: explicitly confirm synthetic test content')
    if not isinstance(filename,str) or len(filename)>255 or '\0' in filename or Path(filename).suffix.lower() not in ('.md','.txt'):
        raise ValueError('Only UTF-8 .txt and .md synthetic documents are supported')
    if not isinstance(content,bytes) or not content or len(content)>262144:
        raise ValueError('Document must contain 1–262144 UTF-8 bytes')
    try:
        body=content.decode('utf-8')
    except UnicodeError:
        raise ValueError('Document is not UTF-8 text') from None
    if not body.strip() or any(ord(char)<32 and char not in '\n\r\t' for char in body):
        raise ValueError('Empty or binary/control-character documents are rejected')
    sha=hashlib.sha256(content).hexdigest()
    with transaction(db) as conn:
        if not conn.execute('SELECT 1 FROM invoice_lines WHERE tenant_id=?',(tenant,)).fetchone():
            raise ValueError('Unknown synthetic tenant')
        prior=conn.execute('SELECT * FROM source_stages WHERE tenant_id=? AND sha256=?',(tenant,sha)).fetchone()
        if prior:
            return dict(prior)
        result={'stage_id':'stage:'+uuid.uuid4().hex,'tenant_id':tenant,'original_filename':filename,
                'body':body,'sha256':sha,'created_at':now()}
        conn.execute('INSERT INTO source_stages VALUES (?,?,?,?,?,?)',tuple(result.values()))
        return result


def list_stages(db,tenant):
    with connect(db) as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM source_stages WHERE tenant_id=? ORDER BY created_at DESC',(tenant,))]


def admit_document(db,tenant,stage_id,scope,title,signed_date,status,synthetic_confirmed,
                   reviewer,note,idempotency_key,replaces_doc_id=None,signed_simulation_confirmed=False):
    _reviewer(reviewer,note)
    if synthetic_confirmed is not True:
        raise ValueError('SYNTHETIC_ONLY: admission requires synthetic confirmation')
    if status not in ('SIGNED','DRAFT') or (status=='SIGNED' and signed_simulation_confirmed is not True):
        raise ValueError('Signed admission requires explicit simulated-signature/scope confirmation')
    signed_date=_date(signed_date,'signed date',status=='DRAFT')
    _require_text(title,'Document title')
    payload=['admit',stage_id,scope,title,signed_date,status,reviewer,note,replaces_doc_id]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        scope=_scope(conn,tenant,scope)
        manifest=synthetic(conn)
        if signed_date and signed_date>manifest['as_of']:
            raise ValueError('Signed date is after the approved snapshot as-of date; use a reviewed snapshot')
        stage=conn.execute('SELECT * FROM source_stages WHERE tenant_id=? AND stage_id=?',(tenant,stage_id)).fetchone()
        if not stage:
            raise ValueError('Staged document not found in the selected tenant')
        duplicate=conn.execute('''SELECT version_id FROM source_versions WHERE tenant_id=?
            AND scope_id=? AND sha256=?''',(tenant,scope_id(scope),stage['sha256'])).fetchone()
        if duplicate:
            return _remember(conn,tenant,idempotency_key,payload,_source(conn,tenant,duplicate[0]))
        previous=None
        if replaces_doc_id:
            row=conn.execute('SELECT version_id FROM source_versions WHERE tenant_id=? AND doc_id=?',
                             (tenant,replaces_doc_id)).fetchone()
            if not row:
                raise ValueError('Previous source not found in this tenant')
            previous=_source(conn,tenant,row[0])
            if previous['scope']!=scope:
                raise ValueError('A source version cannot change its trusted scope')
        version_id='source:'+uuid.uuid4().hex
        doc_id='DOC-'+uuid.uuid4().hex
        result={'version_id':version_id,'tenant_id':tenant,'doc_id':doc_id,
                'family_id':previous['family_id'] if previous else doc_id,
                'scope_id':scope_id(scope),'scope_json':encode(scope),'sha256':stage['sha256'],
                'body':stage['body'],'original_filename':stage['original_filename'],
                'status':status,'signed_date':signed_date,'title':title,'origin':'SYNTHETIC_INTAKE',
                'previous_version':previous['version_id'] if previous else None,'created_at':now()}
        conn.execute('INSERT INTO source_versions VALUES ('+','.join('?' for _ in result)+')',tuple(result.values()))
        doc={'doc_id':doc_id,**scope,'status':status,'kind':'RATE_AGREEMENT','signed_date':signed_date,
             'path':'evidence/'+version_id.replace(':','-')+'.txt','sha256':stage['sha256'],'body':stage['body']}
        names=tuple(doc)
        conn.execute('INSERT INTO documents ('+','.join(names)+') VALUES ('+','.join('?' for _ in names)+')',tuple(doc.values()))
        conn.execute('INSERT INTO document_fts VALUES (?,?,?)',(tenant,doc_id,stage['body']))
        touch_scope(conn,scope,'EVIDENCE_ADMITTED')
        return _remember(conn,tenant,idempotency_key,payload,_source(conn,tenant,version_id))


def dispose_source(db,tenant,version_id,decision,reviewer,note,idempotency_key):
    _reviewer(reviewer,note)
    if decision not in ('NO_PRICING_EFFECT','OUT_OF_SCOPE'):
        raise ValueError('Choose an explicit no-pricing-effect or out-of-scope disposition')
    payload=['source-disposition',version_id,decision,reviewer,note]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        source=_source(conn,tenant,version_id)
        if source['origin']=='CURATED_FIXTURE':
            raise ValueError('Curated baseline source dispositions are preserved')
        if conn.execute("SELECT 1 FROM term_review_events WHERE source_version=? AND decision='TERMS_APPROVED'",(version_id,)).fetchone():
            raise ValueError('Approved pricing evidence cannot be discarded by a source disposition')
        result={'event_id':'source-review:'+uuid.uuid4().hex,'tenant_id':tenant,
                'version_id':version_id,'scope_id':source['scope_id'],'decision':decision,
                'reviewer':reviewer,'note':note,'recorded_at':now()}
        conn.execute('''INSERT INTO source_dispositions(event_id,tenant_id,version_id,scope_id,
            decision,reviewer,note,recorded_at) VALUES (?,?,?,?,?,?,?,?)''',tuple(result.values()))
        touch_scope(conn,source['scope'],decision)
        return _remember(conn,tenant,idempotency_key,payload,result)


def request_for_review(conn,event,request_type='CONTRACT_RATE',details=None):
    """Called in the same transaction as the case decision; stable replay identity."""
    from .legacy_archive import assert_not_archived
    assert_not_archived(conn, event['tenant_id'], event['line_id'])
    if not active(conn):
        return None
    if request_type not in ('CONTRACT_RATE','PAYMENT','CREDIT','SERVICE','OTHER'):
        raise ValueError('Unsupported evidence request type')
    prior=conn.execute('SELECT * FROM evidence_requests WHERE tenant_id=? AND review_event_id=?',
                       (event['tenant_id'],event['event_id'])).fetchone()
    if prior:
        if prior['request_type']!=request_type or prior['details']!=(details or event['note']).strip():
            raise ValueError('This review already has a different structured evidence request')
        return dict(prior)
    assessment=conn.execute('''SELECT assessment_id FROM assessment_revisions WHERE tenant_id=?
        AND line_id=? AND case_digest=? ORDER BY sequence DESC LIMIT 1''',
        (event['tenant_id'],event['line_id'],event['case_digest'])).fetchone()
    result={'request_id':'request:'+digest([event['tenant_id'],event['event_id']]),
            'tenant_id':event['tenant_id'],'line_id':event['line_id'],'review_event_id':event['event_id'],
            'assessment_id':assessment[0] if assessment else None,'request_type':request_type,
            'details':_require_text(details or event['note'],'Missing evidence details'),
            'status':'OPEN','created_at':now()}
    conn.execute('INSERT INTO evidence_requests VALUES (?,?,?,?,?,?,?,?,?)',tuple(result.values()))
    return result


def create_request(db,tenant,review_event_id,request_type,details,idempotency_key):
    payload=['request',review_event_id,request_type,details]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        event=conn.execute('SELECT * FROM review_events WHERE tenant_id=? AND event_id=?',
                           (tenant,review_event_id)).fetchone()
        if not event or event['decision']!='NEEDS_MORE_EVIDENCE':
            raise ValueError('An evidence request must link to a request-more review in this tenant')
        result=request_for_review(conn,dict(event),request_type,details)
        return _remember(conn,tenant,idempotency_key,payload,result)


def list_requests(db,tenant,line_id=None):
    with connect(db) as conn:
        records=[dict(row) for row in conn.execute('SELECT * FROM evidence_requests WHERE tenant_id=? ORDER BY created_at DESC',(tenant,)) if line_id is None or row['line_id']==line_id]
        for record in records:
            record['events']=[dict(row) for row in conn.execute('''SELECT * FROM evidence_request_events
                WHERE tenant_id=? AND request_id=? ORDER BY recorded_at,event_id''',(tenant,record['request_id']))]
            record['attachments']=[row for row in record['events'] if row['decision']=='EVIDENCE_ATTACHED']
        return records


def _request(conn,tenant,request_id):
    row=conn.execute('SELECT * FROM evidence_requests WHERE tenant_id=? AND request_id=?',(tenant,request_id)).fetchone()
    if not row:
        raise ValueError('Evidence request not found in this tenant')
    return dict(row)


def attach_evidence(db,tenant,request_id,version_id,reviewer,note,idempotency_key):
    _reviewer(reviewer,note)
    payload=['attach',request_id,version_id,reviewer,note]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        request=_request(conn,tenant,request_id)
        if request['status']=='RESOLVED':
            raise ValueError('Resolved request cannot accept new attachments')
        source=_source(conn,tenant,version_id)
        line=one_line(conn,tenant,request['line_id'])
        if source['scope']!=scope_of(line):
            raise ValueError('Evidence does not match the request rate scope')
        result={'event_id':'attachment:'+uuid.uuid4().hex,'tenant_id':tenant,
                'request_id':request_id,'version_id':version_id,'decision':'EVIDENCE_ATTACHED',
                'reviewer':reviewer,'note':note,'recorded_at':now()}
        conn.execute('INSERT INTO evidence_request_events VALUES (?,?,?,?,?,?,?,?)',tuple(result.values()))
        conn.execute("UPDATE evidence_requests SET status='EVIDENCE_ATTACHED' WHERE request_id=?",(request_id,))
        return _remember(conn,tenant,idempotency_key,payload,result)


def resolve_request(db,tenant,request_id,reviewer,note,idempotency_key):
    _reviewer(reviewer,note)
    payload=['resolve',request_id,reviewer,note]
    with transaction(db) as conn:
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        request=_request(conn,tenant,request_id)
        if request['request_type']!='CONTRACT_RATE':
            raise ValueError('Only the validated contract/rate evidence path can resolve requests in this milestone')
        versions=[row[0] for row in conn.execute('''SELECT DISTINCT version_id FROM evidence_request_events
            WHERE tenant_id=? AND request_id=? AND decision='EVIDENCE_ATTACHED' ''',(tenant,request_id))]
        if not versions:
            raise ValueError('Attach accepted evidence before resolving the request')
        for version in versions:
            source=_source(conn,tenant,version)
            docs=[dict(row) for row in conn.execute('SELECT * FROM documents WHERE tenant_id=? AND doc_id=?',(tenant,source['doc_id']))]
            reviewed,ignored=coverage(conn,docs)
            curated=source['origin']=='CURATED_FIXTURE' and conn.execute('''SELECT 1 FROM reviewed_terms
                WHERE tenant_id=? AND doc_id=? AND reviewed='true' AND reviewer!='' ''',(tenant,source['doc_id'])).fetchone()
            if not curated and source['doc_id'] not in reviewed|ignored:
                raise ValueError('Attachment needs complete human term review or an explicit source disposition')
        result={'event_id':'request-review:'+uuid.uuid4().hex,'tenant_id':tenant,
                'request_id':request_id,'version_id':None,'decision':'RESOLVED',
                'reviewer':reviewer,'note':note,'recorded_at':now()}
        conn.execute('INSERT INTO evidence_request_events VALUES (?,?,?,?,?,?,?,?)',tuple(result.values()))
        conn.execute("UPDATE evidence_requests SET status='RESOLVED' WHERE request_id=?",(request_id,))
        return _remember(conn,tenant,idempotency_key,payload,result)


def list_assessments(db,tenant,line_id=None):
    from .engine import evaluate_line
    with connect(db) as conn:
        rows=[dict(row) for row in conn.execute('''SELECT * FROM assessment_revisions WHERE tenant_id=?
            ORDER BY sequence DESC''',(tenant,)) if line_id is None or row['line_id']==line_id]
        seen=set()
        current={}
        for row in rows:
            lid=row['line_id']
            if lid not in current:
                current[lid]=evaluate_line(conn,tenant,lid)['case_digest']
            row['case_json']=json.loads(row['case_json'])
            row['validity']='CURRENT' if lid not in seen and row['case_digest']==current[lid] else 'STALE'
            seen.add(lid)
        return rows


def reassess(db,tenant,line_id,prior_review_event_id=None,idempotency_key=None):
    from .engine import evaluate_line
    payload=['reassess',line_id,prior_review_event_id]
    with transaction(db) as conn:
        from .legacy_archive import assert_not_archived
        assert_not_archived(conn, tenant, line_id)
        prior=_previous(conn,tenant,idempotency_key,payload)
        if prior:
            return prior
        if prior_review_event_id:
            event=conn.execute('SELECT * FROM review_events WHERE tenant_id=? AND line_id=? AND event_id=?',
                               (tenant,line_id,prior_review_event_id)).fetchone()
            if not event:
                raise ValueError('Prior review does not belong to this tenant and invoice')
        result=save_assessment(conn,evaluate_line(conn,tenant,line_id),prior_review_event_id=prior_review_event_id)
        return _remember(conn,tenant,idempotency_key,payload,result)


def start_linked_review(db,checkpoint_path,tenant,assessment_id,prior_thread_id=None,
                        idempotency_key=None,use_llm=True,provider='ollama',model='llama3.2:latest',
                        base_url='http://127.0.0.1:11434',context_tokens=16384,retry_preparation=False,
                        retrieval_index=None):
    """Persist intent before starting the separate checkpoint DB; retry never resumes a completed run."""
    from .engine import evaluate_line
    from .workflow import get_review_workflow,run_persistent
    from .lifecycle_storage import review_start_lock
    config={'use_llm':use_llm,'provider':provider,'model':model,'base_url':base_url,'context_tokens':context_tokens}
    if retrieval_index is not None:
        config['retrieval_index'] = retrieval_index
    payload=['start-review',assessment_id,prior_thread_id,config]
    with transaction(db) as conn:
        intent=_previous(conn,tenant,idempotency_key,payload)
        if not intent:
            assessment=conn.execute('SELECT * FROM assessment_revisions WHERE tenant_id=? AND assessment_id=?',(tenant,assessment_id)).fetchone()
            if not assessment:
                raise ValueError('Assessment not found in the selected tenant')
            case=evaluate_line(conn,tenant,assessment['line_id'])
            latest=conn.execute('''SELECT assessment_id FROM assessment_revisions WHERE tenant_id=?
                AND line_id=? ORDER BY sequence DESC LIMIT 1''',(tenant,assessment['line_id'])).fetchone()[0]
            if case['case_digest']!=assessment['case_digest'] or latest!=assessment_id or case['status']!='CANDIDATE':
                raise ValueError('Only the latest current eligible candidate can start a linked review')
            if prior_thread_id and not conn.execute('''SELECT 1 FROM review_events WHERE tenant_id=?
                AND line_id=? AND thread_id=?''',(tenant,case['line_id'],prior_thread_id)).fetchone():
                raise ValueError('Prior thread is not a recorded review for this invoice/tenant')
            intent={'intent_id':'intent:'+uuid.uuid4().hex,'tenant_id':tenant,'line_id':case['line_id'],
                    'assessment_id':assessment_id,'thread_id':f'ui:{time.time_ns():020d}:{uuid.uuid4().hex}',
                    'prior_thread_id':prior_thread_id,'config_json':encode(config),'status':'PENDING','created_at':now()}
            conn.execute('INSERT INTO review_intents VALUES (?,?,?,?,?,?,?,?,?)',tuple(intent.values()))
            _remember(conn,tenant,idempotency_key,payload,intent)
    with review_start_lock(db,intent['thread_id']):
        current=get_review_workflow(db,checkpoint_path,intent['thread_id'],tenant=tenant,line_id=intent['line_id'])
        unfinished = current and current['next'] and current['next']!=['approval'] and not current['state'].get('outcome')
        if current is None or (retry_preparation and unfinished):
            with transaction(db) as conn:
                assessment=conn.execute('SELECT * FROM assessment_revisions WHERE tenant_id=? AND assessment_id=?',
                                        (tenant,intent['assessment_id'])).fetchone()
                case=evaluate_line(conn,tenant,intent['line_id'])
                latest=conn.execute('''SELECT assessment_id FROM assessment_revisions WHERE tenant_id=?
                    AND line_id=? ORDER BY sequence DESC LIMIT 1''',(tenant,intent['line_id'])).fetchone()[0]
                if (not assessment or assessment['case_digest']!=case['case_digest']
                        or latest!=intent['assessment_id'] or case['status']!='CANDIDATE'):
                    raise ValueError('STALE_CASE: review-start intent no longer matches the current eligible assessment')
                expected=assessment['case_digest']
            if current is None:
                current=run_persistent(db,checkpoint_path,intent['thread_id'],tenant=tenant,
                    line_id=intent['line_id'],expected_case_digest=expected,**config)
            else:
                from .workflow import retry_preparation as retry
                current=retry(db,checkpoint_path,intent['thread_id'],tenant=tenant,line_id=intent['line_id'])
        # A retry after checkpoint creation returns that saved state without regeneration.
        with transaction(db) as conn:
            conn.execute("UPDATE review_intents SET status='STARTED' WHERE intent_id=?",(intent['intent_id'],))
        return current
