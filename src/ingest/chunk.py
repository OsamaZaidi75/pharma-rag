"""Chunk label sections into retrieval-friendly pieces.

Strategy: keep each SPL section intact when it fits; split long sections on
sentence boundaries with character overlap. Every chunk carries the drug name
and section title so the generator can cite precisely.
"""
import re
from dataclasses import dataclass

from .spl_parser import LabelSection

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"\u201c])")


@dataclass
class Chunk:
    drug_name: str
    setid: str
    section_code: str
    section_title: str
    chunk_index: int
    content: str


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts or [text.strip()]


def chunk_section(
    section: LabelSection,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    header = f"Drug: {section.drug_name} | Section: {section.section_title}\n"
    body = section.text.strip()
    if len(header) + len(body) <= max_chars:
        return [
            Chunk(
                drug_name=section.drug_name,
                setid=section.setid,
                section_code=section.section_code,
                section_title=section.section_title,
                chunk_index=0,
                content=header + body,
            )
        ]

    chunks: list[Chunk] = []
    sentences = _sentences(body)
    current: list[str] = []
    current_len = len(header)

    def flush(idx: int) -> None:
        chunks.append(
            Chunk(
                drug_name=section.drug_name,
                setid=section.setid,
                section_code=section.section_code,
                section_title=section.section_title,
                chunk_index=idx,
                content=header + " ".join(current),
            )
        )

    idx = 0
    for sent in sentences:
        # Extremely long single sentence: hard-split it.
        while len(sent) > max_chars - len(header):
            cut = max_chars - len(header)
            piece, sent = sent[:cut], sent[cut:]
            if current:
                flush(idx)
                idx += 1
                current, current_len = [], len(header)
            chunks.append(
                Chunk(
                    drug_name=section.drug_name,
                    setid=section.setid,
                    section_code=section.section_code,
                    section_title=section.section_title,
                    chunk_index=idx,
                    content=header + piece,
                )
            )
            idx += 1
        if current_len + len(sent) + 1 > max_chars:
            flush(idx)
            idx += 1
            # Overlap: carry trailing sentences covering ~overlap_chars.
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) >= overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap
            current_len = len(header) + sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        flush(idx)
    return chunks


def chunk_sections(
    sections: list[LabelSection],
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    out: list[Chunk] = []
    for section in sections:
        out.extend(chunk_section(section, max_chars, overlap_chars))
    return out
