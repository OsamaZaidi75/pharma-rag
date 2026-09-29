"""Parse DailyMed Structured Product Label (SPL) XML into sections.

SPL is HL7 v3 XML: sections live under structuredBody, each with a LOINC
<code code="...">, a <title>, and a <text> element. Nested subsections are
extracted as their own records so chunks stay topically coherent.
"""
from dataclasses import dataclass, field
import re

from lxml import etree

NS = {"v3": "urn:hl7-org:v3"}
_SECTION_NUM = re.compile(r"^[\d.]+\s+")

# LOINC -> friendly section name for the sections we care about most.
SECTION_NAMES = {
    "34066-1": "Boxed Warning",
    "34067-9": "Indications and Usage",
    "34068-7": "Dosage and Administration",
    "34069-5": "Dosage Forms and Strengths",
    "34070-3": "Contraindications",
    "34071-1": "Warnings",
    "34072-9": "Warnings and Precautions",
    "34073-7": "Drug Interactions",
    "34075-2": "Use in Specific Populations",
    "34076-0": "Pregnancy",
    "34077-8": "Nursing Mothers",
    "34078-6": "Pediatric Use",
    "34079-4": "Geriatric Use",
    "34080-2": "Overdosage",
    "34081-0": "Description",
    "34082-8": "Clinical Pharmacology",
    "34083-6": "Mechanism of Action",
    "34084-4": "Adverse Reactions",
    "34085-1": "Clinical Studies",
    "34086-9": "How Supplied",
    "34087-7": "Patient Counseling Information",
    "51945-4": "Principal Display Panel",
}


@dataclass
class LabelSection:
    drug_name: str
    setid: str
    section_code: str
    section_title: str
    text: str
    level: int = 0


def _clean_text(text_el: etree._Element) -> str:
    """Extract readable text from a <text> element, skipping nested sections."""
    parts: list[str] = []
    for node in text_el.iter():
        if node.tag == f"{{{NS['v3']}}}section":
            # nested subsections are handled separately; skip their text here
            if node is not text_el.getparent():
                continue
        if node.tag in (
            f"{{{NS['v3']}}}paragraph",
            f"{{{NS['v3']}}}caption",
            f"{{{NS['v3']}}}item",
            f"{{{NS['v3']}}}td",
            f"{{{NS['v3']}}}th",
        ):
            t = "".join(node.itertext()).strip()
            if t:
                parts.append(t)
    # Fallback: if no structured children matched, take all text.
    if not parts:
        t = "".join(text_el.itertext()).strip()
        if t:
            parts.append(t)
    # De-duplicate while preserving order, collapse whitespace.
    seen, out = set(), []
    for p in parts:
        p = " ".join(p.split())
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n".join(out)


def _walk_sections(
    parent: etree._Element,
    drug_name: str,
    setid: str,
    level: int,
    breadcrumb: str,
    out: list[LabelSection],
) -> None:
    for section in parent.findall("v3:component/v3:section", NS):
        code_el = section.find("v3:code", NS)
        code = code_el.get("code", "") if code_el is not None else ""
        title_el = section.find("v3:title", NS)
        raw_title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        # Prefer the label's own title (stripped of numbering like "5.1 ");
        # fall back to the LOINC name when the title is missing. Subsections
        # often reuse the parent's LOINC code, so the code alone would hide
        # the specific title.
        title = _SECTION_NUM.sub("", raw_title).strip() or SECTION_NAMES.get(
            code, "Untitled Section"
        )
        full_title = f"{breadcrumb} > {title}" if breadcrumb else title

        text_el = section.find("v3:text", NS)
        text = _clean_text(text_el) if text_el is not None else ""
        if text and len(text) > 40:  # skip empty/placeholder sections
            out.append(
                LabelSection(
                    drug_name=drug_name,
                    setid=setid,
                    section_code=code,
                    section_title=full_title,
                    text=text,
                    level=level,
                )
            )
        # Recurse into nested subsections.
        _walk_sections(section, drug_name, setid, level + 1, full_title, out)


def parse_spl(xml_bytes: bytes, drug_name: str, setid: str) -> list[LabelSection]:
    """Parse a full SPL document into a flat list of sections."""
    root = etree.fromstring(xml_bytes)
    body = root.find(".//v3:structuredBody", NS)
    if body is None:
        raise ValueError("No structuredBody found - not a valid SPL document")
    sections: list[LabelSection] = []
    _walk_sections(body, drug_name, setid, 0, "", sections)
    return sections


def parse_spl_file(path: str, drug_name: str, setid: str) -> list[LabelSection]:
    with open(path, "rb") as f:
        return parse_spl(f.read(), drug_name, setid)
