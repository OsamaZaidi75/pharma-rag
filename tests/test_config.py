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
    # Same for the cache dir under test below.
    monkeypatch.delenv("SPL_CACHE_DIR", raising=False)


def test_readme_documents_a_default_model():
    with open(README_PATH, encoding="utf-8") as f:
        text = f.read()
    assert MODEL_RE.search(text), "README must state the default model"


def test_gemini_model_default_matches_readme():
    with open(README_PATH, encoding="utf-8") as f:
        text = f.read()
    documented = MODEL_RE.search(text).group(1)
    assert Settings().gemini_model == documented


def test_spl_cache_dir_default_pinned():
    # The on-demand DailyMed path hands settings.spl_cache_dir straight to
    # download_spl_xml(); pin the default here so drift is caught as a test
    # failure, not as a confusing failure at index time.
    assert Settings().spl_cache_dir == "./data/spl_cache"


def test_spl_cache_dir_creatable_and_writable(tmp_path, monkeypatch):
    # Guard the confusing failure mode: the default cache dir must resolve to
    # a location the process can create and write to. Runs in an isolated cwd
    # so the real repo tree is never touched.
    monkeypatch.chdir(tmp_path)
    cache_dir = Settings().spl_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    probe = os.path.join(cache_dir, ".write-probe")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(probe)
    assert os.path.isdir(cache_dir)
