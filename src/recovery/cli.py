from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from .branding import PRODUCT_NAME
from .engine import audit, evaluate_line
from .storage import ImportRejected, connect, load_snapshot
from .retrieval import retrieve_evidence, deterministic_brief


def _import_report_target(args):
    if args.command != 'init' or not args.report:
        return None
    target = Path(args.report)
    try:
        resolved = target.resolve()
        invalid = (resolved == Path(args.db).resolve()
                   or resolved.is_relative_to(Path(args.data).resolve())
                   or target.is_dir() or target.is_symlink()
                   or (target.exists() and (not target.is_file() or target.stat().st_nlink > 1))
                   or any(part.lower().startswith('.env') for part in target.parts))
    except (OSError, RuntimeError, ValueError):
        invalid = True
    if invalid:
        raise ImportRejected({'schema_version': 1, 'status': 'REJECTED', 'reconciliation': None,
            'errors': [{'code': 'REPORT_PATH_ERROR', 'file': 'report', 'row': None, 'field': None,
                        'message': 'Report must be a separate file outside source inputs and the database.'}]})
    return target


def _write_import_report(target, report):
    if target is None:
        return True
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return True
    except OSError:
        report['report_errors'] = [{'code': 'REPORT_WRITE_ERROR', 'file': 'report',
            'row': None, 'field': None,
            'message': 'Could not save the report file; status describes the actual database import outcome.'}]
        return False


def _explanation_options(args):
    return {'provider': getattr(args, 'provider', None) or 'openai',
            'model': getattr(args, 'model', None),
            'base_url': (args.ollama_url if getattr(args, 'ollama_url', None) is not None
                         else 'http://127.0.0.1:11434'),
            'context_tokens': (args.context_tokens
                               if getattr(args, 'context_tokens', None) is not None else 16384)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=PRODUCT_NAME + ' — local synthetic pilot')
    parser.add_argument('--db', default=os.getenv('RECOVERY_DB', 'var/recovery.sqlite3'))
    parser.add_argument('--tenant', default='DEMO', help='Demo scope selector; NOT authentication')
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--data', default='data')
    init.add_argument('--report', help='Write the successful import or structured rejection JSON')
    migration = sub.add_parser('migrate', help='Back up and add the synthetic evidence lifecycle schema')
    migration.add_argument('--backup-dir')
    scan = sub.add_parser('audit')
    scan.add_argument('--output')
    index = sub.add_parser('index-evidence', help='Build a synthetic local Qdrant evidence index')
    index.add_argument('--index', required=True, help='Separate local index directory')
    index.add_argument('--embedding-model', required=True, help='Explicit installed local embedding model')
    index.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    index.add_argument('--max-chars', type=int, default=1200)
    index.add_argument('--overlap-chars', type=int, default=150)
    search = sub.add_parser('search-evidence', help='Search chunks while retaining complete signed evidence')
    search.add_argument('--index', required=True)
    search.add_argument('--line', required=True)
    search.add_argument('--query', required=True)
    search.add_argument('--limit', type=int, default=8, help='Ranked chunk hits; never a document completeness limit')
    show = sub.add_parser('case')
    show.add_argument('--line', required=True)
    explain = sub.add_parser('explain')
    explain.add_argument('--line', required=True)
    explain.add_argument('--llm', action='store_true')
    proposal = sub.add_parser('propose-terms')
    proposal.add_argument('--doc', required=True)
    proposal.add_argument('--output', required=True)
    start = sub.add_parser('graph-start')
    start.add_argument('--line', required=True)
    start.add_argument('--llm', action='store_true')
    resume = sub.add_parser('graph-resume')
    resume.add_argument('--decision', choices=['approve', 'reject', 'request_more'], required=True)
    resume.add_argument('--reviewer', required=True)
    resume.add_argument('--note', required=True)
    resume.add_argument('--evidence-type', choices=['CONTRACT_RATE','PAYMENT','CREDIT','SERVICE','OTHER'],
                        default='CONTRACT_RATE', help='Missing-evidence category for request_more')
    for p in (start, resume):
        p.add_argument('--thread', required=True)
        p.add_argument('--checkpoints', default='var/checkpoints.sqlite3')
    for p in (explain, start):
        p.add_argument('--provider', choices=['openai', 'ollama', 'groq'])
        p.add_argument('--model', help='Explanation model; required for Ollama; hosted Llama default for Groq')
        p.add_argument('--ollama-url', help='Loopback Ollama HTTP endpoint')
        p.add_argument('--context-tokens', type=int, help='Ollama context window (default: 16384)')
        p.add_argument('--retrieval-index', help='Opt-in local semantic evidence index; lexical default otherwise')
    for p in (index, search, explain, start):
        p.add_argument('--allow-local-embedding', action='store_true',
                       help='Explicitly allow this synthetic evidence/query to local Ollama embeddings')
    args = parser.parse_args(argv)
    if args.command in ('index-evidence', 'search-evidence'):
        if not args.allow_local_embedding:
            parser.error('Local embedding requires --allow-local-embedding')
    elif args.command in ('explain', 'graph-start'):
        if bool(args.retrieval_index) != args.allow_local_embedding:
            parser.error('--retrieval-index and --allow-local-embedding must be selected together')
    if args.command in ('explain', 'graph-start') and not args.llm:
        if any(getattr(args, key) is not None
               for key in ('provider', 'model', 'ollama_url', 'context_tokens')):
            parser.error('--provider, --model, --ollama-url and --context-tokens require --llm')
    if args.command in ('explain', 'graph-start') and args.llm:
        if ((args.ollama_url is not None or args.context_tokens is not None)
                and args.provider != 'ollama'):
            parser.error('--ollama-url and --context-tokens require --provider ollama')
    report_target = None
    try:
        report_target = _import_report_target(args)
        if args.command == 'init':
            result = load_snapshot(args.db, args.data)
        elif args.command == 'migrate':
            from .lifecycle import migrate
            result = migrate(args.db, backup_dir=args.backup_dir)
        elif args.command in ('index-evidence', 'search-evidence'):
            from .semantic_retrieval import build_index, rank_evidence, resolve_index_config
            if not Path(args.db).is_file():
                raise ValueError('Initialize an existing synthetic database before indexing or search')
            with connect(args.db) as conn:
                if args.command == 'index-evidence':
                    result = build_index(conn, args.tenant, args.index, model=args.embedding_model,
                        base_url=args.ollama_url, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
                else:
                    documents = retrieve_evidence(conn, args.tenant, args.line, args.query)
                    result = rank_evidence(conn, args.tenant, args.line, documents,
                        resolve_index_config(args.index), query=args.query, limit=args.limit)
        elif args.command.startswith('graph-'):
            from .workflow import run_persistent
            decision = ({'decision': args.decision, 'reviewer': args.reviewer, 'note': args.note}
                        if args.command == 'graph-resume' else None)
            if decision is not None:
                decision['evidence_type'] = args.evidence_type
            result = run_persistent(args.db, args.checkpoints, args.thread, tenant=args.tenant,
                line_id=getattr(args, 'line', None), use_llm=getattr(args, 'llm', False),
                decision=decision, retrieval_index=getattr(args, 'retrieval_index', None),
                **_explanation_options(args))
        else:
            with connect(args.db) as conn:
                if args.command == 'audit':
                    result = audit(conn, args.tenant)
                elif args.command == 'case':
                    result = evaluate_line(conn, args.tenant, args.line)
                elif args.command == 'explain':
                    case = evaluate_line(conn, args.tenant, args.line)
                    retrieval_config = None
                    if args.retrieval_index is not None:
                        from .semantic_retrieval import resolve_index_config
                        retrieval_config = resolve_index_config(args.retrieval_index)
                    documents = retrieve_evidence(conn, args.tenant, args.line,
                                                   retrieval_config=retrieval_config)
                    if args.llm:
                        from .ai import explain_case, resolve_explanation_config
                        config = resolve_explanation_config(**_explanation_options(args))
                        brief = explain_case(case, documents, **config)
                    else:
                        brief = deterministic_brief(case)
                    result = {'case': case, 'brief': brief,
                              'retrieved_doc_ids': [d['doc_id'] for d in documents]}
                else:
                    row = conn.execute('SELECT * FROM documents WHERE tenant_id=? AND doc_id=?',
                                       (args.tenant, args.doc)).fetchone()
                    if row is None:
                        raise ValueError('Document not found in this tenant')
                    from .ai import propose_terms
                    result = propose_terms(dict(row))
        if args.command == 'init' and not _write_import_report(report_target, result):
            print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
            raise SystemExit(2)
        text = json.dumps(result, indent=2, ensure_ascii=False)
        output = getattr(args, 'output', None)
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(text + '\n', encoding='utf-8')
            print(f'Wrote {output}')
        if args.command == 'audit':
            print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
        else:
            print(text)
        return 0
    except ImportRejected as exc:
        _write_import_report(report_target, exc.report)
        print(json.dumps(exc.report, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc
    except (ValueError, RuntimeError, ImportError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
