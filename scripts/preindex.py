#!/usr/bin/env python3
"""Pre-index drugs so the first question about each one is instant.

Same on-demand path a first question would trigger (DailyMed download +
embed + upsert into ChromaDB), run up front — e.g. at deploy/startup.
Already-indexed drugs are skipped, so re-running is safe.

Usage:
    python scripts/preindex.py                                   # default drugs
    python scripts/preindex.py --drugs atorvastatin metformin    # pick your own

Default drugs: atorvastatin, metformin, lisinopril.

Each drug takes ~30-60s on first index. Embeddings go through the NVIDIA
API (needs NVIDIA_API_KEY). The index lives in CHROMA_DIR (default
./data/chroma) — mount it as a volume in Docker so pre-indexing survives
container restarts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.ingest.ondemand import ensure_drug_indexed  # noqa: E402
from src.store import vectorstore  # noqa: E402

DEFAULT_DRUGS = ["atorvastatin", "metformin", "lisinopril"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-index drugs from DailyMed")
    parser.add_argument("--drugs", nargs="*", default=DEFAULT_DRUGS,
                        help="Generic drug names to index")
    args = parser.parse_args()

    vectorstore.init_db()
    print(f"Embedding model (NVIDIA API): {settings.embedding_model}")
    embedder = Embedder(settings.embedding_model)

    for drug in args.drugs:
        known = {d.lower() for d in vectorstore.list_drugs()}
        if drug.lower() in known:
            print(f"[{drug}] already indexed, skipping")
            continue
        print(f"[{drug}] indexing from DailyMed...")
        try:
            got = ensure_drug_indexed(drug, embedder)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        if got:
            print(f"  indexed {drug}")
        else:
            print(f"  no DailyMed label found for {drug}")

    print(f"Indexed drugs: {vectorstore.list_drugs()}")
    print(f"Total chunks in DB: {vectorstore.count_chunks()}")


if __name__ == "__main__":
    main()
