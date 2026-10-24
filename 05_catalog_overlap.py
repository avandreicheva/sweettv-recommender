"""
How much of the weeks 31-42 labelled catalog was already seen in weeks 11-30:
  - distinct tv_show_id overlap (show-level "have we ever seen this show before")
  - row-level overlap (how many of the 31-42 *airings* are of an already-seen show)
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
CHUNKSIZE = 500_000

PAST = [DATA_DIR / "export_arh_11-20-final.csv", DATA_DIR / "export_arh_21-30-final.csv"]
FUTURE = DATA_DIR / "export_arh_31-42-final.csv"


def labelled_show_ids(path):
    ids = set()
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, usecols=["tv_show_id"]):
        c = chunk[chunk["tv_show_id"] != 0]
        ids.update(c["tv_show_id"].unique().tolist())
    return ids


def main():
    past_ids = set()
    for p in PAST:
        past_ids |= labelled_show_ids(p)
    print(f"distinct labelled tv_show_id in weeks 11-30: {len(past_ids)}")

    future_rows = 0
    future_ids = set()
    overlap_rows = 0
    for chunk in pd.read_csv(FUTURE, chunksize=CHUNKSIZE, usecols=["tv_show_id"]):
        lab = chunk[chunk["tv_show_id"] != 0]
        future_rows += len(lab)
        future_ids.update(lab["tv_show_id"].unique().tolist())
        overlap_rows += int(lab["tv_show_id"].isin(past_ids).sum())

    show_overlap = future_ids & past_ids
    print(f"distinct labelled tv_show_id in weeks 31-42: {len(future_ids)}")
    print(f"distinct shows in both (31-42 ∩ 11-30): {len(show_overlap)} "
          f"({100*len(show_overlap)/len(future_ids):.1f}% of 31-42 catalog)")
    print(f"labelled airings (rows) in weeks 31-42: {future_rows}")
    print(f"  of which the show was already seen in weeks 11-30: {overlap_rows} "
          f"({100*overlap_rows/future_rows:.1f}%)")


if __name__ == "__main__":
    main()
