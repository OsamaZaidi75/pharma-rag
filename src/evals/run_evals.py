"""Retrieval evals against a golden Q&A set.

Metrics:
- hit@k: fraction of questions where a chunk from the expected drug AND
  expected section appears in the top-k retrieved chunks.
- MRR: mean reciprocal rank of the first correct chunk.

Run: python -m src.evals.run_evals [--no-rerank]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import settings  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden.json")


def is_hit(chunks, expected_drug: str, expected_section: str) -> int | None:
    """Return 1-based rank of first chunk matching drug+section, else None."""
    exp_drug = expected_drug.lower()
    exp_sec = expected_section.lower()
    for rank, c in enumerate(chunks, start=1):
        if exp_drug in c.drug_name.lower() and exp_sec in c.section_title.lower():
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    retriever = Retriever(
        embedder=Embedder(settings.embedding_model),
        use_reranker=not args.no_rerank,
    )

    hits, reciprocal_ranks = 0, []
    print(f"{'question':60s} {'rank':>5s}  hit?")
    print("-" * 75)
    for item in golden:
        chunks = retriever.retrieve(item["question"], top_k=args.top_k, top_n=args.top_k)
        rank = is_hit(chunks, item["expected_drug"], item["expected_section"])
        hit = rank is not None
        hits += hit
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        mark = "Y" if hit else "X"
        print(f"{item['question'][:58]:60s} {str(rank):>5s}  {mark}")

    n = len(golden)
    print("-" * 75)
    print(f"hit@{args.top_k}: {hits}/{n} = {hits / n:.2%}")
    print(f"MRR:        {sum(reciprocal_ranks) / n:.3f}")
    print(f"reranker:   {'off' if args.no_rerank else 'on'}")


if __name__ == "__main__":
    main()
