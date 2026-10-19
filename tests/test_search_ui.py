"""Search suggestions remain optional filters, without automatic selection."""
from __future__ import annotations

import unittest

from recovery.search_ui import autocomplete_search


class SearchWidget:
    def __init__(self, state=None, selected=None):
        self.session_state = state or {}
        self.selected = selected
        self.calls = []

    def selectbox(self, label, options, **kwargs):
        self.calls.append((label, options, kwargs))
        return self.selected


class AutocompleteSearchTests(unittest.TestCase):
    def test_empty_query_has_no_implicit_first_selection(self):
        widget = SearchWidget()
        self.assertEqual(autocomplete_search(widget, 'Search vendor', ['SUP-A', 'SUP-B'], key='q'), '')
        _, options, configuration = widget.calls[0]
        self.assertEqual(options, ['SUP-A', 'SUP-B'])
        self.assertIsNone(configuration['index'])
        self.assertTrue(configuration['accept_new_options'])

    def test_raw_candidates_are_normalized_deduplicated_without_composite_labels(self):
        widget = SearchWidget(selected='SUP-A')
        actual = autocomplete_search(widget, 'Search',
            ['SUP-A', None, '', '  ', ' AsterTech ', 'SUP-A', 123], key='q')
        self.assertEqual(actual, 'SUP-A')
        self.assertEqual(widget.calls[0][1], ['SUP-A', 'AsterTech'])

    def test_custom_substring_and_no_match_queries_are_retained(self):
        for query in ('astertech', 'no such vendor', '  partial name  '):
            with self.subTest(query=query):
                widget = SearchWidget({'q': query}, selected=query)
                self.assertEqual(autocomplete_search(widget, 'Search', ['SUP-A'], key='q'), query.strip())
                self.assertIn(query, widget.calls[0][1])

    def test_empty_options_still_accept_free_text_and_custom_placeholder(self):
        widget = SearchWidget(selected='custom search')
        self.assertEqual(autocomplete_search(widget, 'Search', [], key='q',
            placeholder='Ask the workspace'), 'custom search')
        self.assertEqual(widget.calls[0][2]['placeholder'], 'Ask the workspace')


if __name__ == '__main__':
    unittest.main()
