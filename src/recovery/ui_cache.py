"""Bounded display caching; decisions always use the uncached review services."""
from __future__ import annotations

from pathlib import Path

_cached_worklist = None


def portfolio_stamp(db, checkpoints):
    """Observe the SQLite files and WALs without creating/opening either DB."""
    result = []
    for database in (Path(db), Path(checkpoints)):
        for path in (database, database.with_name(database.name + '-wal')):
            try:
                stat = path.stat()
            except FileNotFoundError:
                result.append((str(path.resolve()), None))
            else:
                # Opening a checkpoint reader may create an empty WAL. It
                # contains no committed state and is equivalent to no WAL.
                if path.name.endswith('-wal') and stat.st_size == 0:
                    result.append((str(path.resolve()), None))
                else:
                    result.append((str(path.resolve()), stat.st_dev, stat.st_ino,
                                   stat.st_mtime_ns, stat.st_size))
    return tuple(result)


def _read_worklist(db, checkpoints, tenant, stamp):
    from .invoice_review import load_worklist

    result = load_worklist(db, checkpoints, tenant=tenant)
    if portfolio_stamp(db, checkpoints) != stamp:
        raise RuntimeError('The portfolio changed while loading. Refresh the page before reviewing.')
    return result


def display_worklist(st, db, checkpoints, *, tenant='DEMO'):
    """Cache a display snapshot only; file/WAL changes invalidate it immediately.

    Frozen UI selection tokens still face fresh, uncached case/attempt checks at
    every decision. This cache never supplies authority to an approval service.
    """
    global _cached_worklist
    if _cached_worklist is None:
        _cached_worklist = st.cache_data(show_spinner=False, ttl=60, max_entries=8)(_read_worklist)
    result = _cached_worklist(str(Path(db).resolve()), str(Path(checkpoints).resolve()), tenant,
                              portfolio_stamp(db, checkpoints))
    # A rebuilt/deleted semantic index is outside the two SQLite files. Recheck
    # saved semantic readiness on each display, without embedding/model calls.
    from .invoice_review import decision_error
    for case in result['cases']:
        if (case.get('workflow') or {}).get('state', {}).get('retrieval_config') is not None:
            case['decision_error'] = decision_error(db, case, tenant=tenant)
            case['ready'] = case['decision_error'] is None
    return result
