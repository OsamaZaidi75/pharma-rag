"""Unit tests for the ReAct step parser. No LLM needed — pure function."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.react import parse_step


def test_parse_answer():
    step = parse_step("ANSWER: The label lists headache and nausea.")
    assert step.kind == "answer"
    assert "headache" in step.summary


def test_parse_action_with_thought():
    step = parse_step(
        "I need label evidence first.\n"
        'ACTION: search_labels\n'
        'ARGS: {"question": "atorvastatin side effects", "top_n": 5}'
    )
    assert step.kind == "action"
    assert step.tool_name == "search_labels"
    assert step.args == {"question": "atorvastatin side effects", "top_n": 5}
    assert "label evidence" in step.thought


def test_parse_action_no_thought():
    step = parse_step('ACTION: list_indexed_drugs\nARGS: {}')
    assert step.kind == "action"
    assert step.tool_name == "list_indexed_drugs"
    assert step.args == {}


def test_parse_bad_json_raises():
    with pytest.raises(ValueError):
        parse_step('ACTION: search_labels\nARGS: {not valid json}')


def test_parse_neither_format_raises():
    with pytest.raises(ValueError):
        parse_step("Hmm, interesting question, let me think about it.")


def test_parse_unknown_tool_name_still_parses():
    # Unknown tools are rejected at execution time, not parse time.
    step = parse_step('ACTION: teleport\nARGS: {"x": 1}')
    assert step.kind == "action"
    assert step.tool_name == "teleport"
