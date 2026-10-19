from __future__ import annotations

import copy
from datetime import date
import hashlib
import unittest
import uuid

from recovery.chunking import CHUNKER_VERSION, chunk_document


def document(body='Signed amendment: an invented approved hourly rate applies.', **changes):
    result = {'tenant_id': 'DEMO', 'doc_id': 'DOC-TEST', 'body': body,
              'sha256': hashlib.sha256(body.encode('utf-8')).hexdigest()}
    result.update(changes)
    return result


class ChunkingTests(unittest.TestCase):
    def assert_coverage(self, source, chunks, *, limit, overlap):
        body = source['body']
        covered_until = 0
        previous_start = -1
        for index, chunk in enumerate(chunks):
            start, end = chunk['character_start'], chunk['character_end']
            self.assertGreater(start, previous_start)
            self.assertLessEqual(start, covered_until)
            self.assertGreater(end, start)
            self.assertLessEqual(end - start, limit)
            self.assertEqual(chunk['text'], body[start:end])
            self.assertEqual(chunk['source_sha256'], source['sha256'])
            if index:
                self.assertEqual(start, chunks[index - 1]['character_end'] - overlap)
            covered_until = max(covered_until, end)
            previous_start = start
        self.assertEqual(chunks[0]['character_start'], 0)
        self.assertEqual(covered_until, len(body))
        self.assertEqual(len({chunk['chunk_id'] for chunk in chunks}), len(chunks))

    def test_short_document_has_exact_source_and_stable_uuid_v5(self):
        source = document()
        chunks = chunk_document(source)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], source['body'])
        self.assertEqual(chunks[0]['character_start'], 0)
        self.assertEqual(chunks[0]['character_end'], len(source['body']))
        self.assertEqual(chunks[0]['line_start'], 1)
        self.assertEqual(chunks[0]['line_end'], 1)
        self.assertEqual(chunks[0]['chunker_version'], CHUNKER_VERSION)
        self.assertEqual(uuid.UUID(chunks[0]['chunk_id']).version, 5)
        self.assertEqual(chunks, chunk_document(source))

    def test_long_unbroken_clause_has_full_coverage_and_exact_overlap(self):
        source = document('Clause Ω⚖：' + 'é合同🙂' * 800)
        chunks = chunk_document(source, max_chars=128, overlap_chars=23)
        self.assertGreater(len(chunks), 20)
        self.assert_coverage(source, chunks, limit=128, overlap=23)
        self.assertTrue(all(chunk['line_start'] == chunk['line_end'] == 1 for chunk in chunks))

    def test_paragraph_boundary_preferred_over_later_single_newline(self):
        source = document('a' * 48 + '\n\n' + 'b' * 7 + '\n' + 'c' * 60)
        chunks = chunk_document(source, max_chars=64, overlap_chars=10)
        self.assertEqual(chunks[0]['character_end'], 50)
        self.assertEqual(chunks[0]['text'], 'a' * 48 + '\n\n')
        self.assert_coverage(source, chunks, limit=64, overlap=10)

    def test_newline_in_latter_half_is_used_before_hard_cut(self):
        source = document('a' * 40 + '\n' + 'b' * 70)
        chunks = chunk_document(source, max_chars=64, overlap_chars=0)
        self.assertEqual(chunks[0]['character_end'], 41)
        self.assertEqual(chunks[1]['line_start'], 2)
        self.assert_coverage(source, chunks, limit=64, overlap=0)

    def test_early_boundary_does_not_create_tiny_or_stalled_windows(self):
        source = document('a\n\n' + 'b' * 400)
        chunks = chunk_document(source, max_chars=64, overlap_chars=31)
        self.assertEqual(chunks[0]['character_end'], 64)
        self.assert_coverage(source, chunks, limit=64, overlap=31)

    def test_odd_window_half_boundary_cannot_stall_at_maximum_overlap(self):
        source = document('a' * 30 + '\n\n' + 'b' * 400)
        chunks = chunk_document(source, max_chars=65, overlap_chars=32)
        self.assertEqual(chunks[0]['character_end'], 65)
        self.assert_coverage(source, chunks, limit=65, overlap=32)

    def test_crlf_blank_paragraph_is_kept_and_counts_as_one_break(self):
        source = document('a' * 40 + '\r\n\r\n' + 'b' * 100)
        chunks = chunk_document(source, max_chars=64, overlap_chars=0)
        self.assertEqual(chunks[0]['character_end'], 44)
        self.assertEqual(chunks[0]['line_start'], 1)
        self.assertEqual(chunks[0]['line_end'], 2)
        self.assertEqual(chunks[1]['line_start'], 3)
        self.assert_coverage(source, chunks, limit=64, overlap=0)

    def test_single_crlf_is_not_misclassified_as_blank_paragraph(self):
        source = document('a' * 38 + '\r\n' + 'b' * 15 + '\r\n' + 'c' * 80)
        chunks = chunk_document(source, max_chars=64, overlap_chars=0)
        self.assertEqual(chunks[0]['character_end'], 57)
        self.assertEqual(chunks[0]['line_end'], 2)
        self.assertEqual(chunks[1]['line_start'], 3)
        self.assert_coverage(source, chunks, limit=64, overlap=0)

    def test_hard_boundary_does_not_separate_crlf(self):
        source = document('a' * 63 + '\r\n' + 'b' * 100)
        chunks = chunk_document(source, max_chars=64, overlap_chars=0)
        self.assertEqual(chunks[0]['character_end'], 63)
        self.assertTrue(chunks[1]['text'].startswith('\r\n'))
        self.assertEqual(chunks[1]['line_start'], 1)
        self.assert_coverage(source, chunks, limit=64, overlap=0)

    def test_unicode_offsets_are_characters_not_utf8_bytes(self):
        body = 'é' * 40 + '\n' + '合同🙂' * 60
        source = document(body)
        chunks = chunk_document(source, max_chars=64, overlap_chars=7)
        self.assertEqual(chunks[0]['character_end'], 41)
        self.assertEqual(len(chunks[0]['text'].encode('utf-8')), 81)
        self.assert_coverage(source, chunks, limit=64, overlap=7)

    def test_all_whitespace_and_line_terminators_remain_in_source(self):
        source = document('\t  Original heading\r\n\r\n' + ('  clause\t \r\n' * 30) + '  tail  \n')
        chunks = chunk_document(source, max_chars=80, overlap_chars=13)
        self.assertTrue(chunks[0]['text'].startswith('\t  '))
        self.assertTrue(chunks[-1]['text'].endswith('  tail  \n'))
        self.assert_coverage(source, chunks, limit=80, overlap=13)

    def test_last_included_newline_belongs_to_its_original_line(self):
        chunks = chunk_document(document('first\r\nsecond\nthird\r'))
        self.assertEqual(chunks[0]['line_start'], 1)
        self.assertEqual(chunks[0]['line_end'], 3)

    def test_metadata_and_signed_date_ordinal_are_preserved_without_input_mutation(self):
        metadata = {'supplier_id': 'SUP-A', 'contract_id': 'CTR-A', 'role': 'ENGINEER',
            'location': 'REMOTE', 'currency': 'USD', 'unit': 'HOUR', 'service_type': 'REGULAR',
            'status': 'SIGNED', 'kind': 'RATE_AGREEMENT', 'signed_date': '2026-07-01',
            'path': 'contracts/DOC-TEST.md'}
        source = document(**metadata)
        before = copy.deepcopy(source)
        chunk = chunk_document(source)[0]
        self.assertEqual(source, before)
        for key, value in metadata.items():
            self.assertEqual(chunk[key], value)
        self.assertEqual(chunk['signed_date_ordinal'], date(2026, 7, 1).toordinal())

    def test_optional_scope_and_date_are_not_invented(self):
        chunk = chunk_document(document())[0]
        self.assertNotIn('supplier_id', chunk)
        self.assertNotIn('signed_date', chunk)
        self.assertNotIn('signed_date_ordinal', chunk)

    def test_identity_changes_with_tenant_document_source_and_settings(self):
        source = document('content' * 60)
        original = chunk_document(source, max_chars=80, overlap_chars=10)[0]['chunk_id']
        variants = [
            chunk_document(document(source['body'], tenant_id='OTHER'), max_chars=80, overlap_chars=10),
            chunk_document(document(source['body'], doc_id='DOC-OTHER'), max_chars=80, overlap_chars=10),
            chunk_document(document(source['body'] + '!'), max_chars=80, overlap_chars=10),
            chunk_document(source, max_chars=81, overlap_chars=10),
            chunk_document(source, max_chars=80, overlap_chars=11),
        ]
        for chunks in variants:
            self.assertNotEqual(chunks[0]['chunk_id'], original)

    def test_invalid_required_fields_are_rejected(self):
        invalid = [None, [], 'text', {}, document(tenant_id=''), document(doc_id=' '),
            document(tenant_id=' DEMO'), document(doc_id='bad\nid'), document() | {'body': None},
            document() | {'body': 12}, document(body=''), document(body=' \r\n\t'),
            document(sha256=None), document(sha256='0' * 63), document(sha256='x' * 64),
            document(sha256='0' * 64), document(sha256='A' * 64)]
        for source in invalid:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    chunk_document(source)

    def test_body_with_unpaired_surrogate_is_rejected(self):
        source = document() | {'body': 'broken\ud800Unicode', 'sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'UTF-8'):
            chunk_document(source)

    def test_invalid_window_and_overlap_are_rejected(self):
        for size in (True, False, '1200', None, 64.0, 0, 63, 12001):
            with self.subTest(max_chars=size):
                with self.assertRaises(ValueError):
                    chunk_document(document(), max_chars=size)
        for overlap in (True, False, '1', None, 1.0, -1, 32, 64):
            with self.subTest(overlap_chars=overlap):
                with self.assertRaises(ValueError):
                    chunk_document(document(), max_chars=64, overlap_chars=overlap)

    def test_boundaries_of_permitted_window_sizes_execute(self):
        source = document('x' * 25000)
        for limit, overlap in ((64, 31), (65, 32), (12000, 5999), (12000, 0)):
            with self.subTest(limit=limit, overlap=overlap):
                chunks = chunk_document(source, max_chars=limit, overlap_chars=overlap)
                self.assert_coverage(source, chunks, limit=limit, overlap=overlap)

    def test_invalid_calendar_signed_date_is_rejected(self):
        for signed_date in ('2026-02-30', '20260202', '', None, 20260701, '2026-07-01T00:00:00'):
            with self.subTest(signed_date=signed_date):
                with self.assertRaisesRegex(ValueError, 'calendar date'):
                    chunk_document(document(signed_date=signed_date))


if __name__ == '__main__':
    unittest.main()
