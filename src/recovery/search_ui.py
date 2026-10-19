"""Native typeahead search with optional free-text filtering.

Suggestions are display aids supplied from the caller's existing record scope.
They never select a row, approve a case or change a database record.
"""
from __future__ import annotations


def autocomplete_search(st, label, values, *, key,
                        placeholder='Type to search or choose a suggestion', format_func=str):
    """Suggest known raw search values while retaining arbitrary substrings.

    Keep the original search key so deep links and filter-reset callbacks retain
    their existing behavior. A blank field means no filter, never the first
    suggestion. Accepted custom values remain available on the next rerun.
    """
    options = list(dict.fromkeys(value.strip() for value in values
                                if isinstance(value, str) and value.strip()))
    current = st.session_state.get(key)
    if isinstance(current, str) and current.strip() and current not in options:
        options.append(current)
    selected = st.selectbox(label, options, index=None, key=key,
        accept_new_options=True, placeholder=placeholder, format_func=format_func,
        help='Choose a matching suggestion, or enter your own search. Clear the field to show all matches.')
    return selected.strip() if isinstance(selected, str) else ''
