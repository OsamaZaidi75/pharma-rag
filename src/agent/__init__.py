"""ReAct agent: the model plans its own tool calls instead of following a fixed pipeline.

The fixed RAG pipeline (retrieve -> rerank -> generate) is one good path, but
it can't adapt: it can't decide "I need another drug's label first" or "let me
check what's indexed before searching". The agent loop hands those decisions
to the model:

    think -> act (call a tool) -> observe result -> repeat until done

See react.py for the loop, tools.py for the tool registry.
"""
