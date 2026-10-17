"""Smoke test: the pinned gemini_model default must match the README's
documented model.

Guards the stale-model drift seen with the 2.5-flash deprecation wave — if the
config default and the README disagree about the model, this test fails.
The README documents the default as:  Default model: `gemini-3.6-flash`
"""
import os
import re

import pytest

from src.config import Settings

README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")
MODEL_RE = re.compile(r"Default model:\s*`([^`]+)`")


@pytest.fixture(autouse=True)
def _clean_model_env(monkeypatch):
    # An ambient GEMINI_MODEL would override the pinned default under test.
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


def test_readme_documents_a_default_model():
    with open(README_PATH, encoding="utf-8") as f:
        text = f.read()
    assert MODEL_RE.search(text), "README must state the default model"


def test_gemini_model_default_matches_readme():
    with open(README_PATH, encoding="utf-8") as f:
        text = f.read()
    documented = MODEL_RE.search(text).group(1)
    assert Settings().gemini_model == documented
