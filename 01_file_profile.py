"""
Per-file structural profile, streamed in chunks (no full-file load).
Reports: rows, columns, start_time min/max + format, set of ISO weeks covered.
"""
import sys
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
CHUNKSIZE = 500_000

FILES = {
    "dataset11-30.csv": {"ts_col": "start_time", "ts_fmt": "%Y-%m-%d %H:%M:%S"},
    "export_arh_11-20-final.csv": {"ts_col": "start_time", "ts_fmt": "%d.%m.%Y %H:%M:%S"},
    "export_arh_21-30-final.csv": {"ts_col": "start_time", "ts_fmt": "%d.%m.%Y %H:%M:%S"},
    "export_arh_31-42-final.csv": {"ts_col": "start_time", "ts_fmt": "%d.%m.%Y %H:%M:%S"},
    "submission.csv": None,  # no timestamp column
}


def profile_timestamped(path: Path, ts_col: str, ts_fmt: str):
    rows = 0
    cols = None
    tmin = None
    tmax = None
    weeks = set()
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, usecols=[ts_col]):
        rows += len(chunk)
        if cols is None:
            cols = None  # filled below from full header read
        ts = pd.to_datetime(chunk[ts_col], format=ts_fmt)
        cmin, cmax = ts.min(), ts.max()
        tmin = cmin if tmin is None or cmin < tmin else tmin
        tmax = cmax if tmax is None or cmax > tmax else tmax
        wk = ts.dt.isocalendar().week.unique()
        weeks.update(wk.tolist())
    # read header separately (cheap) for column list
    header = pd.read_csv(path, nrows=0)
    return rows, list(header.columns), tmin, tmax, sorted(weeks)


def profile_plain(path: Path):
    header = pd.read_csv(path, nrows=0)
    rows = 0
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, usecols=[header.columns[0]]):
        rows += len(chunk)
    return rows, list(header.columns)


def main():
    for fname, spec in FILES.items():
        path = DATA_DIR / fname
        print(f"\n=== {fname} ===")
        if spec is None:
            rows, cols = profile_plain(path)
            print(f"rows={rows} cols={cols}")
            continue
        rows, cols, tmin, tmax, weeks = profile_timestamped(path, spec["ts_col"], spec["ts_fmt"])
        print(f"rows={rows}")
        print(f"cols={cols}")
        print(f"start_time format='{spec['ts_fmt']}' min={tmin} max={tmax}")
        print(f"ISO weeks covered ({len(weeks)}): {weeks}")


if __name__ == "__main__":
    main()
