#!/usr/bin/env python3
"""Download FDA drug labels (SPL XML) from DailyMed.

Usage:
    python scripts/download_labels.py --drugs atorvastatin metformin lisinopril
    python scripts/download_labels.py --drugs atorvastatin --out data/labels --max-per-drug 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import dailymed  # noqa: E402

DEFAULT_DRUGS = ["atorvastatin", "metformin", "lisinopril", "amlodipine", "omeprazole"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SPL XML labels from DailyMed")
    parser.add_argument("--drugs", nargs="*", default=DEFAULT_DRUGS)
    parser.add_argument("--out", default="data/labels")
    parser.add_argument("--max-per-drug", type=int, default=1,
                        help="How many labels (newest first) to keep per drug")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for drug in args.drugs:
        print(f"[{drug}] searching DailyMed...")
        try:
            spls = dailymed.search_spls(drug, pagesize=10)
        except Exception as e:
            print(f"  search failed: {e}")
            continue
        if not spls:
            print("  no labels found")
            continue
        for spl in spls[: args.max_per_drug]:
            fname = f"{drug.replace(' ', '_')}_{spl.setid}.xml"
            path = os.path.join(args.out, fname)
            if os.path.exists(path):
                print(f"  already have {fname}")
                continue
            print(f"  downloading {spl.title[:60]}...")
            try:
                xml = dailymed.download_spl_xml(spl.setid)
            except Exception as e:
                print(f"  download failed: {e}")
                continue
            with open(path, "wb") as f:
                f.write(xml)
            total += 1
            print(f"  saved {fname} ({len(xml) // 1024} KB)")
    print(f"Done. {total} new label(s) in {args.out}/")


if __name__ == "__main__":
    main()
