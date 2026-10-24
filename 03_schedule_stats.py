"""
Streaming stats for the three schedule files:
- total programmes (rows)
- share with tv_show_id != 0 ("labelled")
- category value counts
"""
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent
CHUNKSIZE = 500_000

FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
]


def main():
    for fname in FILES:
        path = DATA_DIR / fname
        rows = 0
        labelled = 0
        cat_counts = Counter()
        for chunk in pd.read_csv(
            path,
            chunksize=CHUNKSIZE,
            usecols=["tv_show_id", "tv_show_category"],
        ):
            rows += len(chunk)
            labelled += int((chunk["tv_show_id"] != 0).sum())
            cat_counts.update(chunk["tv_show_category"].fillna("<NA>").value_counts().to_dict())

        print(f"\n=== {fname} ===")
        print(f"programmes (rows)={rows}")
        print(f"labelled (tv_show_id != 0)={labelled} ({100*labelled/rows:.1f}%)")
        print("categories (count, share):")
        for cat, cnt in cat_counts.most_common():
            print(f"  {cat:30s} {cnt:7d}  {100*cnt/rows:5.1f}%")


if __name__ == "__main__":
    main()
