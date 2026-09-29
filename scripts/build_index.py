#!/usr/bin/env python3
"""Parse -> chunk -> embed -> index. Run after download_labels.py.

Usage:
    python scripts/build_index.py                      # index data/labels
    python scripts/build_index.py --rebuild              # drop and rebuild everything
    python scripts/build_index.py --labels-dir data/more_labels
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm  # noqa: E402

from src.config import settings  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.ingest.chunk import chunk_sections  # noqa: E402
from src.ingest.spl_parser import parse_spl_file  # noqa: E402
from src.store import vectorstore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pharma RAG index")
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop all indexed data first")
    args = parser.parse_args()

    if args.rebuild:
        print("Dropping existing index...")
        vectorstore.drop_all()
    else:
        vectorstore.init_db()

    xml_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.xml")))
    if not xml_files:
        print(f"No XML files in {args.labels_dir}/. Run download_labels.py first.")
        sys.exit(1)

    # 1. Parse + chunk (fast, no models needed)
    all_chunks = []
    for path in tqdm(xml_files, desc="Parsing"):
        fname = os.path.basename(path)
        # filename format: {drug}_{setid}.xml  (setid has no underscores)
        drug, setid = fname[:-4].rsplit("_", 1)
        try:
            sections = parse_spl_file(path, drug_name=drug, setid=setid)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue
        chunks = chunk_sections(
            sections,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        all_chunks.extend(chunks)
    print(f"Parsed {len(xml_files)} labels -> {len(all_chunks)} chunks")

    # 2. Embed (loads the model once)
    print(f"Loading embedding model: {settings.embedding_model}")
    embedder = Embedder(settings.embedding_model)
    texts = [c.content for c in all_chunks]
    embeddings = []
    batch = 64
    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        embeddings.extend(embedder.embed(texts[i: i + batch]))

    # 3. Upsert
    setids = {c.setid for c in all_chunks}
    n = vectorstore.upsert_chunks(
        all_chunks, embeddings,
        delete_setids=setids,
    )
    print(f"Indexed {n} chunks from {len(setids)} label(s).")
    print(f"Total chunks in DB: {vectorstore.count_chunks()}")


if __name__ == "__main__":
    main()
