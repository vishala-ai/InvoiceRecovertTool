"""Optional LangGraph orchestration with SQLite checkpoints and an explicit approval interrupt."""
from __future__ import annotations

from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3
from typing import TypedDict

from .engine import evaluate_line
from .retrieval import retrieve_evidence, deterministic_brief
from .review import record_review
from .storage import connect


class CaseState(TypedDict, total=False):
    tenant: str
    line_id: str
    thread_id: str
    use_llm: bool
    explanation_config: dict
    case: dict
    documents: list[dict]
    brief: dict
    outcome: dict
    expected_case_digest: str
    assessment_id: str
    retrieval_config: dict


def _pinned_explanation_config(state):
    config = state.get('explanation_config')
    required = {'provider', 'model', 'base_url', 'context_tokens', 'model_digest', 'prompt_version'}
    if (not isinstance(config, dict) or set(config) != required
            or not isinstance(config.get('provider'), str)
            or config.get('provider') not in {'openai', 'ollama', 'groq'}
            or any(not isinstance(config.get(key), str) or not config[key].strip()
                   for key in ('model', 'base_url', 'prompt_version'))
            or type(config.get('context_tokens')) is not int or config['context_tokens'] <= 0
            or (config['model_digest'] is not None
                and not isinstance(config['model_digest'], str))
            or (config['provider'] == 'ollama'
                and (not isinstance(config.get('model_digest'), str)
                     or not config['model_digest'].strip()))):
        raise ValueError('Unpinned LLM workflow: start a new thread with an explicit provider/model')
    from .ai import PROMPT_VERSION
    if config['prompt_version'] != PROMPT_VERSION:
        raise ValueError('Explanation prompt version changed; start a new thread and review')
    if config['provider'] == 'groq':
        from .groq_provider import BASE_URL
        if config['base_url'] != BASE_URL or config['model_digest'] is not None:
            raise ValueError('Unpinned Groq workflow: start a new thread with the hosted endpoint')
    return config


def _validate_cloud_state(state, *, completion_db=None):
    """Reject local providers/indexes before a restored Cloud node can contact them."""
    if os.getenv('RECOVERY_CLOUD') != '1':
        return
    if completion_db is not None:
        # A completed journal write may need its interrupted checkpoint finished
        # after server settings change. This path invokes no provider; the journal
        # service still compares the complete original decision payload on replay.
        with connect(completion_db) as conn:
            committed = conn.execute('''SELECT 1 FROM review_events WHERE event_id=?
                AND thread_id=? AND tenant_id=? AND line_id=? AND case_digest=?''',
                (state['thread_id'] + ':review', state['thread_id'], state['tenant'],
                 state['line_id'], state['case']['case_digest'])).fetchone()
        if committed is not None:
            return
    if state.get('retrieval_config') is not None:
        raise ValueError('Cloud reviews use complete canonical evidence without local embeddings')
    if state.get('use_llm'):
        from .groq_provider import DEFAULT_MODEL
        config = _pinned_explanation_config(state)
        if (config['provider'] != 'groq'
                or config['model'] != os.getenv('GROQ_MODEL', DEFAULT_MODEL)):
            raise ValueError('Cloud explanation settings changed; prepare a fresh hosted review')


def build_graph(db_path, checkpointer):
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import interrupt
    except ImportError as exc:
        raise RuntimeError('Install graph dependencies: uv sync --extra graph') from exc

    def assess(state):
        _validate_cloud_state(state)
        with connect(db_path) as conn:
            from .lifecycle_storage import active, save_assessment
            if active(conn):
                from .lifecycle import _previous, _remember
                key = 'workflow-assessment:' + state['thread_id']
                payload = ['workflow-assessment',state['tenant'],state['line_id']]
                with conn:
                    conn.execute('BEGIN IMMEDIATE')
                    case = evaluate_line(conn, state['tenant'], state['line_id'])
                    if state.get('expected_case_digest', case['case_digest']) != case['case_digest']:
                        raise ValueError('STALE_CASE: linked assessment changed before workflow start')
                    previous = _previous(conn,state['tenant'],key,payload)
                    if previous and previous['case_digest'] != case['case_digest']:
                        raise ValueError('STALE_CASE: start a fresh review after input changes')
                    if not previous:
                        # A linked intent already owns its immutable assessment revision.
                        linked = conn.execute('SELECT assessment_id FROM review_intents WHERE thread_id=?',
                                              (state['thread_id'],)).fetchone()
                        if linked:
                            row = conn.execute('SELECT * FROM assessment_revisions WHERE assessment_id=?',
                                               (linked[0],)).fetchone()
                            record = {'assessment_id': row['assessment_id'], 'case_digest': row['case_digest']}
                        else:
                            record = save_assessment(conn,case)
                        _remember(conn,state['tenant'],key,payload,record)
                    else:
                        record = previous
            else:
                case = evaluate_line(conn, state['tenant'], state['line_id'])
            return {'case': case, **({'assessment_id':record['assessment_id']} if active(conn) else {})}

    def retrieve(state):
        _validate_cloud_state(state)
        with connect(db_path) as conn:
            return {'documents': retrieve_evidence(conn, state['tenant'], state['line_id'],
                retrieval_config=state.get('retrieval_config'))}

    def draft(state):
        _validate_cloud_state(state)
        with connect(db_path) as conn:
            if state.get('retrieval_config') is not None:
                from .semantic_retrieval import validate_index_config
                validate_index_config(conn, state['tenant'], state['retrieval_config'])
            if evaluate_line(conn,state['tenant'],state['line_id'])['case_digest'] != state['case']['case_digest']:
                raise ValueError('STALE_CASE: source inputs changed before explanation')
        if state.get('use_llm'):
            from .ai import explain_case
            brief = explain_case(state['case'], state['documents'],
                                 **_pinned_explanation_config(state))
        else:
            brief = deterministic_brief(state['case'])
        with connect(db_path) as conn:
            if evaluate_line(conn,state['tenant'],state['line_id'])['case_digest'] != state['case']['case_digest']:
                raise ValueError('STALE_CASE: late explanation withheld after input changes')
        return {'brief': brief}

    def approve(state):
        _validate_cloud_state(state, completion_db=db_path)
        event_id = state['thread_id'] + ':review'
        if state.get('retrieval_config') is not None:
            from .semantic_retrieval import validate_index_config
            with connect(db_path) as conn:
                committed = conn.execute('''SELECT 1 FROM review_events WHERE event_id=?
                    AND thread_id=? AND tenant_id=? AND line_id=? AND case_digest=?''',
                    (event_id, state['thread_id'], state['tenant'], state['line_id'],
                     state['case']['case_digest'])).fetchone()
                # A node can fail after its journal commit but before checkpointing
                # its outcome. Finishing that exact historical event must not need
                # an index that may since have been rebuilt or removed. The journal
                # API below still checks the complete decision payload on replay.
                if committed is None:
                    validate_index_config(conn, state['tenant'], state['retrieval_config'])
        if state.get('use_llm'):
            _pinned_explanation_config(state)
        # Never put non-idempotent effects before interrupt: the node is re-executed on resume.
        decision = interrupt({'kind': 'human_review', 'case': state['case'], 'draft': state['brief'],
                              'allowed_decisions': ['approve', 'reject', 'request_more']})
        with connect(db_path) as conn:
            event = record_review(conn, event_id=event_id,
                thread_id=state['thread_id'], tenant=state['tenant'], line_id=state['line_id'],
                decision=decision['decision'], reviewer=decision['reviewer'], note=decision['note'],
                case_digest=state['case']['case_digest'],
                evidence_type=decision.get('evidence_type', 'CONTRACT_RATE'),
                assessment_id=state.get('assessment_id'))
        return {'outcome': event}

    builder = StateGraph(CaseState)
    builder.add_node('assess', assess)
    builder.add_node('retrieve', retrieve)
    builder.add_node('draft', draft)
    builder.add_node('approval', approve)
    builder.add_edge(START, 'assess')
    builder.add_edge('assess', 'retrieve')
    builder.add_edge('retrieve', 'draft')
    builder.add_conditional_edges('draft', lambda s: 'approval'
        if s['case']['status'] == 'CANDIDATE' else END)
    builder.add_edge('approval', END)
    return builder.compile(checkpointer=checkpointer)


def _workflow_view(thread_id, snapshot, created_at=None):
    return {'thread_id': thread_id, 'paused': bool(snapshot.next),
            'next': list(snapshot.next), 'state': snapshot.values,
            **({'created_at':created_at} if created_at else {})}


def _attempt_created_at(graph, config):
    # Checkpoint update time is not review-attempt creation time. Read history
    # without invoking nodes, including legacy CLI threads with arbitrary IDs.
    dates=[item.created_at for item in graph.get_state_history(config) if item.created_at]
    return min(dates) if dates else None


@contextmanager
def _existing_checkpoint_saver(checkpoint_path):
    """Read an existing saver without creating a database or migrating its schema."""
    path = Path(checkpoint_path)
    if not path.exists():
        yield None
        return
    if not path.is_file():
        raise ValueError('Checkpoint path must be a file')
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError('Install graph dependencies: uv sync --extra graph') from exc
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True,
                                 check_same_thread=False)) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if 'checkpoints' not in tables:
            yield None
            return
        if 'writes' not in tables:
            raise ValueError('Incomplete workflow checkpoint database')
        saver = SqliteSaver(conn)
        # SqliteSaver's lazy setup issues DDL even during reads. The existing
        # schema is read-only here; support its pre-task_path format as well.
        saver.is_setup = True
        saver._has_task_path = any(row[1] == 'task_path'
                                   for row in conn.execute('PRAGMA table_info(writes)'))
        yield saver


def get_review_workflow(db_path, checkpoint_path, thread_id: str, *,
                        tenant='DEMO', line_id=None) -> dict | None:
    """Load a saved view without invoking nodes, models or review-journal writes."""
    if not thread_id.strip():
        raise ValueError('A nonempty thread ID is required')
    with _existing_checkpoint_saver(checkpoint_path) as saver:
        if saver is None:
            return None
        graph=build_graph(db_path,saver)
        config={'configurable': {'thread_id': thread_id}}
        snapshot = graph.get_state(config)
        if not snapshot.values:
            return None
        if snapshot.values.get('tenant') != tenant:
            raise ValueError('Thread does not belong to the selected tenant')
        if line_id is not None and snapshot.values.get('line_id') != line_id:
            raise ValueError('Thread does not belong to the selected invoice line')
        return _workflow_view(thread_id, snapshot, _attempt_created_at(graph,config))


def list_review_workflows(db_path, checkpoint_path, *,
                          tenant='DEMO', line_id=None) -> list[dict]:
    """Discover saved root workflows in the selected demo tenant/invoice scope."""
    with _existing_checkpoint_saver(checkpoint_path) as saver:
        if saver is None:
            return []
        graph = build_graph(db_path, saver)
        thread_ids = [row[0] for row in saver.conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns='' ORDER BY thread_id")]
        views = []
        for thread_id in thread_ids:
            snapshot = graph.get_state({'configurable': {'thread_id': thread_id}})
            if (snapshot.values.get('tenant') == tenant
                    and (line_id is None or snapshot.values.get('line_id') == line_id)):
                views.append(_workflow_view(thread_id, snapshot, _attempt_created_at(graph,
                    {'configurable':{'thread_id':thread_id}})))
        return views


def run_persistent(db_path, checkpoint_path, thread_id: str, *,
                   tenant='DEMO', line_id=None, use_llm=False, decision=None,
                   provider='openai', model=None, base_url='http://127.0.0.1:11434',
                   context_tokens=16384, expected_case_digest=None, retrieval_index=None) -> dict:
    if os.getenv('RECOVERY_CLOUD') == '1' and decision is None:
        if retrieval_index is not None:
            raise ValueError('Cloud reviews cannot use local embedding indexes')
        if use_llm:
            from .groq_provider import DEFAULT_MODEL
            if provider != 'groq' or model not in (None, os.getenv('GROQ_MODEL', DEFAULT_MODEL)):
                raise ValueError('Cloud reviews require the server-configured Groq model')
    # Migration history has no new review lifecycle. Reject it before creating
    # a checkpoint database or resolving any explanation/embedding provider.
    if line_id is not None:
        from .legacy_archive import assert_not_archived
        with connect(db_path) as conn:
            assert_not_archived(conn, tenant, line_id)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command
    except ImportError as exc:
        raise RuntimeError('Install graph dependencies: uv sync --extra graph') from exc
    if not thread_id.strip():
        raise ValueError('A nonempty thread ID is required')
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    config = {'configurable': {'thread_id': thread_id}}
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_graph(db_path, saver)
        prior = graph.get_state(config)
        if decision is not None:
            if prior.next != ('approval',):
                raise ValueError('No pending human-review interrupt exists for this thread')
            if prior.values.get('tenant') != tenant:
                raise ValueError('Thread does not belong to the selected tenant')
            if prior.values.get('use_llm'):
                _pinned_explanation_config(prior.values)
            _validate_cloud_state(prior.values, completion_db=db_path)
            graph.invoke(Command(resume=decision), config=config)
        else:
            if prior.values:
                raise ValueError('Thread already exists; resume it or choose a new ID')
            if not line_id:
                raise ValueError('A line ID is required')
            initial = {'tenant': tenant, 'line_id': line_id, 'thread_id': thread_id,
                       'use_llm': use_llm}
            if retrieval_index is not None:
                from .semantic_retrieval import resolve_index_config
                initial['retrieval_config'] = resolve_index_config(retrieval_index)
            if expected_case_digest is not None:
                initial['expected_case_digest'] = expected_case_digest
            if use_llm:
                from .ai import resolve_explanation_config
                initial['explanation_config'] = resolve_explanation_config(
                    provider=provider, model=model, base_url=base_url,
                    context_tokens=context_tokens)
            graph.invoke(initial, config=config)
        current = graph.get_state(config)
        return _workflow_view(thread_id, current, _attempt_created_at(graph,config))


def retry_preparation(db_path, checkpoint_path, thread_id, *, tenant, line_id):
    """Explicitly retry an unfinished preparation node; never resume a human/completed node."""
    from .legacy_archive import assert_not_archived
    with connect(db_path) as conn:
        assert_not_archived(conn, tenant, line_id)
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph=build_graph(db_path,saver)
        config={'configurable':{'thread_id':thread_id}}
        before=graph.get_state(config)
        if (before.values.get('tenant')!=tenant or before.values.get('line_id')!=line_id
                or not before.next or before.next==('approval',) or before.values.get('outcome')):
            raise ValueError('Only unfinished preparation for this tenant/line can be retried')
        _validate_cloud_state(before.values)
        graph.invoke(None,config=config)
        return _workflow_view(thread_id,graph.get_state(config),_attempt_created_at(graph,config))
