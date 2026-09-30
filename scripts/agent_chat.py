"""Terminal demo of the ReAct agent — watch it think, act, and observe.

Run:
    python scripts/agent_chat.py "Compare the side effects of atorvastatin and rosuvastatin"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import react


def main() -> None:
    question = (
        " ".join(sys.argv[1:])
        or "Compare the side effects of atorvastatin and rosuvastatin."
    )
    print(f"Question: {question}\n")
    result = react.run_agent(question)

    for step in result.trace:
        print(f"--- step {step['iteration']} ---")
        if step.get("thought"):
            print(f"Thought: {step['thought'][:300]}")
        if step.get("action"):
            print(f"Action:  {step['action']}({step['args']})")
            print(f"Observe: {step['observation'][:500]}")
        else:
            print("(model finished gathering evidence)")
        print()

    print(f"ANSWER (after {result.iterations} steps):\n")
    print(result.answer)
    print(f"\n[model: {result.model}]")


if __name__ == "__main__":
    main()
