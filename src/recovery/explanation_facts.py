"""Closed, auditable explanation claims; no model arithmetic or semantic judge.

Only Python-authored facts can become visible factual prose. This deliberately
does not attempt to certify arbitrary natural-language entailment with regexes.
The evaluation corpus is never an input to this operational module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib

from .retrieval import validate_model_citations

CASE_FIELDS = ('line_id', 'status', 'reason', 'currency', 'billed_amount',
               'expected_amount', 'gross_discrepancy', 'offsets', 'net_candidate',
               'as_of', 'po_rate')
AMOUNTS = {'billed_amount', 'expected_amount', 'gross_discrepancy', 'offsets',
           'net_candidate', 'po_rate'}
FINANCIAL_IDS = {f'case.{key}' for key in (*AMOUNTS, 'status', 'currency', 'realized_recovery')}
STATUSES = {'CANDIDATE', 'NEEDS_REVIEW', 'NO_DISCREPANCY', 'ALREADY_OFFSET', 'UNPAID'}
ACTIONS = {
    'review_candidate': 'Review the supplied service-date terms, settlement and offsets before '
                        'deciding whether to prepare a draft. No supplier action is authorized.',
    'resolve_evidence': 'Resolve the stated evidence or scope issue and run a fresh assessment '
                        'before considering a draft.',
    'retain_assessment': 'Review the documented assessment and retain its non-candidate outcome '
                         'unless new evidence supports a fresh assessment.',
}


class DraftRejected(ValueError):
    """A withheld draft with a safe, structured check report, never its raw prose."""

    def __init__(self, message, validation):
        super().__init__(message)
        self.validation = deepcopy(validation)


def required_action(case):
    return ('review_candidate' if case['status'] == 'CANDIDATE' else
            'resolve_evidence' if case['status'] == 'NEEDS_REVIEW' else 'retain_assessment')


def _string(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Incomplete authoritative explanation facts; draft withheld')
    return value


def _numeric(value, *, nullable=False):
    if value is None and nullable:
        return
    _string(value)
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError('Invalid authoritative financial string; draft withheld') from None
    if not number.is_finite() or number < Decimal('0'):
        raise ValueError('Invalid authoritative financial string; draft withheld')


def _provenance(case, documents):
    """Check the curated case's provenance as well as the model's quote membership."""
    if not isinstance(documents, list) or any(not isinstance(d, dict) for d in documents):
        raise ValueError('Incomplete authoritative evidence; draft withheld')
    by_id = {}
    for document in documents:
        doc_id = _string(document.get('doc_id'))
        body = document.get('body')
        if doc_id in by_id or not isinstance(body, str):
            raise ValueError('Evidence document IDs must be unique and bodies complete')
        digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
        if document.get('sha256') != digest:
            raise ValueError('Evidence source hash mismatch; draft withheld')
        for key in ('tenant_id', 'supplier_id', 'contract_id'):
            if key in document and key in case and document[key] != case[key]:
                raise ValueError('Evidence scope mismatch; draft withheld')
        if (document.get('status', 'SIGNED') != 'SIGNED'
                or document.get('kind', 'RATE_AGREEMENT') != 'RATE_AGREEMENT'
                or document.get('signed_date', case['as_of']) > case['as_of']):
            raise ValueError('Evidence is outside the signed as-of scope; draft withheld')
        by_id[doc_id] = document
    citations = case.get('citations')
    if not isinstance(citations, list):
        raise ValueError('Incomplete authoritative citation facts; draft withheld')
    terms = set()
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError('Invalid authoritative citation; draft withheld')
        document = by_id.get(citation.get('doc_id'))
        if (document is None or not isinstance(citation.get('quote'), str)
                or not citation['quote'] or citation['quote'] not in document['body']
                or ('sha256' in citation and citation['sha256'] != document['sha256'])):
            raise ValueError('Authoritative citation provenance mismatch; draft withheld')
        terms.add((citation.get('term_id'), citation['doc_id']))
    return terms


def build_fact_catalog(case: dict, documents: list[dict]) -> list[dict]:
    """Project the assessed case into mandatory, exact, read-only explanation facts.

    Unknown amounts stay null. No absence of a component/amount is converted to a
    negative claim about the documents. Source quotes remain the reviewed baseline.
    """
    if not isinstance(case, dict) or any(key not in case for key in CASE_FIELDS):
        raise ValueError('Incomplete authoritative explanation facts; draft withheld')
    for key in CASE_FIELDS:
        if key in AMOUNTS:
            _numeric(case[key], nullable=key not in {'billed_amount', 'net_candidate'})
        else:
            _string(case[key])
    if case['status'] not in STATUSES:
        raise ValueError('Unknown authoritative assessment status; draft withheld')
    try:
        date.fromisoformat(case['as_of'])
    except ValueError:
        raise ValueError('Invalid authoritative as-of date; draft withheld') from None
    if case.get('realized_recovery', '0.00') != '0.00':
        raise ValueError('This demo cannot record realized recovery; draft withheld')
    terms = _provenance(case, documents)
    currency = case['currency']
    text = {
        'line_id': f"Invoice line {case['line_id']}.",
        'status': f"Python assessment: {case['status']}.",
        'reason': f"Assessment reason: {case['reason']}.",
        'currency': f'Case currency: {currency}.',
        'as_of': f"Snapshot as of {case['as_of']}.",
    }
    labels = {'billed_amount': 'Billed amount', 'expected_amount': 'Expected amount',
              'gross_discrepancy': 'Gross rate discrepancy', 'offsets': 'Applied pricing offsets',
              'net_candidate': 'Net candidate', 'po_rate': 'Purchase-order hourly rate'}
    for key, label in labels.items():
        text[key] = (f'{label} is unresolved or unassessed; retain the assessment reason.'
                     if case[key] is None else f'{label}: {currency} {case[key]}.')
    if case['po_rate'] is not None:
        text['po_rate'] += ' A PO match alone does not establish the correct contract rate.'
    facts = [{'fact_id': f'case.{key}', 'value': case[key], 'text': text[key]}
             for key in CASE_FIELDS]
    facts.append({'fact_id': 'case.realized_recovery', 'value': '0.00',
                  'text': 'New realized recovery: 0.00. A candidate is not recovered money.'})
    components = case.get('components')
    if not isinstance(components, list):
        raise ValueError('Incomplete authoritative service components; draft withheld')
    seen = set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError('Invalid authoritative service component; draft withheld')
        for key in ('service_entry_id', 'service_date', 'hours', 'rate', 'term_id', 'doc_id'):
            _string(component.get(key))
        _numeric(component['hours'])
        _numeric(component['rate'])
        try:
            date.fromisoformat(component['service_date'])
        except ValueError:
            raise ValueError('Invalid authoritative service date; draft withheld') from None
        entry_id = component['service_entry_id']
        if entry_id in seen or (component['term_id'], component['doc_id']) not in terms:
            raise ValueError('Service component provenance mismatch; draft withheld')
        seen.add(entry_id)
        facts.append({'fact_id': f'component.{entry_id}.rate', 'value': component['rate'],
                      'text': f"Service entry {entry_id}: {component['hours']} approved hours on "
                              f"{component['service_date']} at {currency} {component['rate']}/HOUR "
                              f"under reviewed term {component['term_id']} ({component['doc_id']})."})
    warnings = case.get('warnings', [])
    if not isinstance(warnings, list):
        raise ValueError('Incomplete authoritative warnings; draft withheld')
    for index, warning in enumerate(warnings):
        facts.append({'fact_id': f'warning.{index}', 'value': _string(warning),
                      'text': f'Assessment warning: {warning}'})
    return facts


def evaluate_claims(claims: list[dict], catalog: list[dict]) -> dict:
    """Classify closed claims by exact fact identity/value, without an LLM judge.

    'Supported' means equality to this catalog, not general prose entailment or
    independently verified real-world contractual correctness.
    """
    facts = {fact['fact_id']: fact for fact in catalog}
    if len(facts) != len(catalog):
        raise ValueError('Duplicate authoritative fact IDs')
    results, supported, counts = [], set(), Counter()
    for claim in claims if isinstance(claims, list) else [None]:
        fact_id = claim.get('fact_id') if isinstance(claim, dict) else None
        valid = (isinstance(claim, dict) and set(claim) == {'fact_id', 'value'}
                 and isinstance(fact_id, str))
        if isinstance(fact_id, str):
            counts[fact_id] += 1
        if not valid or fact_id not in facts:
            classification = 'unsupported'
        elif (type(claim['value']) is type(facts[fact_id]['value'])
              and claim['value'] == facts[fact_id]['value']):
            classification = 'supported'
            supported.add(fact_id)
        else:
            classification = 'contradicted'
        results.append({'fact_id': fact_id if isinstance(fact_id, str) else None,
                        'classification': classification})
    missing = sorted(set(facts) - supported)
    duplicate = sorted(key for key, count in counts.items() if count > 1)
    financial = {key for key in facts if key in FINANCIAL_IDS or key.startswith('component.')}
    consistency = all(key in supported and counts[key] == 1 for key in financial)
    passed = not missing and not duplicate and all(
        item['classification'] == 'supported' for item in results)
    return {'status': 'PASSED' if passed else 'WITHHELD', 'claims': results,
            'missing_fact_ids': missing, 'duplicate_fact_ids': duplicate,
            'financial_consistency': consistency}


def validate_fact_draft(draft: dict, case: dict, documents: list[dict]) -> dict:
    """Withhold failed model output and render only validated Python facts."""
    if (not isinstance(draft, dict)
            or set(draft) != {'claims', 'next_action', 'citations'}):
        raise ValueError('Unsupported free prose or invalid fact draft; draft withheld')
    catalog = build_fact_catalog(case, documents)
    validation = evaluate_claims(draft['claims'], catalog)
    validation.update(scope='closed_fact_contract', arbitrary_prose_entailment='NOT_ASSESSED',
                      citation_check='NOT_ESTABLISHED', action_check='NOT_ESTABLISHED')
    if any(c['classification'] != 'supported' for c in validation['claims']):
        raise DraftRejected('Model claims contradicted or unsupported by authoritative facts; '
                            'draft withheld', validation)
    if validation['status'] != 'PASSED':
        raise DraftRejected('Model claims incomplete or duplicated; draft withheld', validation)
    expected_action = required_action(case)
    if draft['next_action'] != expected_action:
        validation.update(status='WITHHELD', action_check='WITHHELD')
        raise DraftRejected('Model next action is unsupported by assessment status; draft withheld',
                            validation)
    validation['action_check'] = 'PASSED'
    if (not isinstance(draft['citations'], list)
            or any(not isinstance(cite, dict) or set(cite) != {'doc_id', 'quote'}
                   or any(not isinstance(cite[key], str) or not cite[key]
                          for key in ('doc_id', 'quote'))
                   for cite in draft['citations'])):
        raise ValueError('Model returned invalid citations; draft withheld')
    try:
        validate_model_citations(draft, documents, case)
    except ValueError as exc:
        validation.update(status='WITHHELD', citation_check='WITHHELD')
        raise DraftRejected(str(exc), validation) from None
    required = {(cite['doc_id'], cite['quote']) for cite in case['citations']}
    actual = {(cite['doc_id'], cite['quote']) for cite in draft['citations']}
    if not required.issubset(actual):
        validation.update(status='WITHHELD', citation_check='WITHHELD')
        raise DraftRejected('Model omitted reviewed governing quotations; draft withheld', validation)
    validation['citation_check'] = 'PASSED'
    limitations = ['Synthetic fixtures only; curated reviewed terms are not automatic extraction.',
                   'Human review of applicability, source completeness and settlement remains required.',
                   'Only validated fact templates are displayed; arbitrary prose entailment is not assessed.',
                   'No supplier action, approval or new realized recovery is produced by this draft.']
    if case['status'] == 'NEEDS_REVIEW':
        limitations.append(f"Unresolved assessment: {case['reason']}. Unknown amounts remain unknown.")
    return {'summary': ' '.join(fact['text'] for fact in catalog),
            'next_action': ACTIONS[expected_action], 'limitations': limitations,
            'citations': deepcopy(draft['citations']), 'source_citations': deepcopy(case['citations']),
            'claims': deepcopy(draft['claims']), 'validation': validation,
            'composition': 'validated_fact_template',
            **{key: case[key] for key in ('case_digest', 'source_digest', 'as_of') if key in case}}
