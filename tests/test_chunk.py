"""Unit tests for SPL parsing + chunking (no models, no DB needed)."""
import os

from src.ingest.chunk import chunk_sections
from src.ingest.spl_parser import parse_spl_file

SAMPLE = os.path.join(os.path.dirname(__file__), "../data/sample/sample_spl.xml")


def test_parse_sections():
    sections = parse_spl_file(SAMPLE, drug_name="testdrug", setid="abc123")
    titles = [s.section_title for s in sections]
    assert any("contraindications" in t.lower() for t in titles)
    assert any("warnings and precautions" in t.lower() for t in titles)
    # Nested subsection extracted separately with breadcrumb title.
    assert any("driving and operating machinery" in t.lower() for t in titles)
    assert all(s.drug_name == "testdrug" for s in sections)
    assert all(len(s.text) > 40 for s in sections)


def test_chunking_short_section_stays_whole():
    sections = parse_spl_file(SAMPLE, drug_name="testdrug", setid="abc123")
    contra = [s for s in sections if "contraindications" in s.section_title.lower()][0]
    chunks = chunk_sections([contra], max_chars=1200, overlap_chars=200)
    assert len(chunks) == 1
    assert "Drug: testdrug" in chunks[0].content
    assert "hypersensitivity" in chunks[0].content


def test_chunking_long_section_splits_with_overlap():
    sections = parse_spl_file(SAMPLE, drug_name="testdrug", setid="abc123")
    warnings = [s for s in sections if s.section_title.lower() == "warnings and precautions"][0]
    chunks = chunk_sections([warnings], max_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    # Headers preserved on every chunk.
    assert all(c.content.startswith("Drug: testdrug") for c in chunks)
    # Chunk indices are sequential.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # No chunk exceeds the limit (with small tolerance for header).
    assert all(len(c.content) <= 700 for c in chunks)


def test_rrf_fuse():
    from src.store.vectorstore import rrf_fuse

    fused = rrf_fuse([[1, 2, 3], [2, 3, 4]])
    ids = [i for i, _ in fused]
    assert ids[0] == 2  # ranked highly by both lists
    assert set(ids) == {1, 2, 3, 4}
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
