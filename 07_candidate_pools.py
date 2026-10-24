"""
Candidate pool per target window = distinct labelled tv_show_id airing in that
window's SCHEDULE file only (never from view logs). Repeats across methods/users
are allowed; the pool itself is a de-duplicated set of show ids.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
CHUNKSIZE = 500_000


def distinct_labelled_ids(path):
    ids = set()
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, usecols=["tv_show_id"]):
        c = chunk[chunk["tv_show_id"] != 0]
        ids.update(c["tv_show_id"].unique().tolist())
    return sorted(ids)


def main():
    pool_val = distinct_labelled_ids(DATA_DIR / "export_arh_21-30-final.csv")
    pool_test = distinct_labelled_ids(DATA_DIR / "export_arh_31-42-final.csv")

    pd.DataFrame({"tv_show_id": pool_val}).to_csv(DATA_DIR / "candidate_pool_VAL.csv", index=False)
    pd.DataFrame({"tv_show_id": pool_test}).to_csv(DATA_DIR / "candidate_pool_TEST.csv", index=False)

    print(f"candidate_pool_VAL.csv:  {len(pool_val)} shows (from export_arh_21-30-final.csv)")
    print(f"candidate_pool_TEST.csv: {len(pool_test)} shows (from export_arh_31-42-final.csv)")


if __name__ == "__main__":
    main()
