"""Edge-case tests for conversational query rewriting (src/generation/rewrite.py)."""
from unittest.mock import MagicMock, patch

from src.generation.rewrite import rewrite_question


def _mock_client(rewritten: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = rewritten
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def test_empty_history_skips_llm():
    """No history at all -> question returned unchanged, LLM never called."""
    with patch("src.generation.rewrite.get_llm_client") as mock_get:
        assert rewrite_question("what about its dosage?", []) == "what about its dosage?"
        mock_get.assert_not_called()


def test_none_history_skips_llm():
    with patch("src.generation.rewrite.get_llm_client") as mock_get:
        assert rewrite_question("side effects?", None) == "side effects?"
        mock_get.assert_not_called()


def test_unusable_history_rows_filtered():
    """History with wrong roles / empty content counts as no history."""
    history = [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "   "},
        {"role": "tool", "content": "some tool output"},
    ]
    with patch("src.generation.rewrite.get_llm_client") as mock_get:
        assert rewrite_question("interactions?", history) == "interactions?"
        mock_get.assert_not_called()


def test_long_history_is_truncated_to_last_10():
    """Long conversations only send the last 10 messages to the LLM."""
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(20)
    ]
    client = _mock_client("What is the dosage of atorvastatin?")
    with patch("src.generation.rewrite.get_llm_client", return_value=(client, "test-model")):
        out = rewrite_question("what about its dosage?", history)
    assert out == "What is the dosage of atorvastatin?"
    # system prompt + 10 history messages + the rewrite instruction
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert len(messages) == 12
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "message 10"
    assert messages[10]["content"] == "message 19"


def test_llm_failure_falls_back_to_original():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    with patch("src.generation.rewrite.get_llm_client", return_value=(client, "test-model")):
        assert rewrite_question("contraindications?", [{"role": "user", "content": "atorvastatin?"}]) == "contraindications?"


def test_empty_llm_output_falls_back_to_original():
    client = _mock_client('   "" ')
    with patch("src.generation.rewrite.get_llm_client", return_value=(client, "test-model")):
        assert rewrite_question("dosage?", [{"role": "user", "content": "atorvastatin?"}]) == "dosage?"


def test_quotes_and_whitespace_stripped():
    client = _mock_client('  "What are the side effects of ibuprofen?"  \n')
    with patch("src.generation.rewrite.get_llm_client", return_value=(client, "test-model")):
        assert rewrite_question("side effects?", [{"role": "user", "content": "ibuprofen?"}]) == "What are the side effects of ibuprofen?"
