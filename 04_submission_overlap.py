"""
For submission.csv users:
  (a) how many appear at all in the view log (dataset11-30.csv)
  (b) how many have at least one viewing session that time-overlaps a
      labelled (tv_show_id != 0) programme on the same channel
        join key: view_log.vsetv_id == schedule.channel_id
        overlap: view.start < prog.stop  AND  view.stop > prog.start
        schedule source: export_arh_11-20-final.csv + export_arh_21-30-final.csv
                          (the two schedule files that cover the view-log period)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
VIEW_PATH = DATA_DIR / "dataset11-30.csv"
SCHED_PATHS = [DATA_DIR / "export_arh_11-20-final.csv", DATA_DIR / "export_arh_21-30-final.csv"]
SUB_PATH = DATA_DIR / "submission.csv"
CHUNKSIZE = 500_000


def load_submission_users():
    sub = pd.read_csv(SUB_PATH, usecols=["user_id"], dtype={"user_id": "int64"})
    return set(sub["user_id"].tolist())


def build_labelled_schedule_index():
    """channel_id -> (start_arr sorted, stop_arr aligned, tv_show_id_arr aligned)"""
    frames = []
    for p in SCHED_PATHS:
        for chunk in pd.read_csv(
            p,
            chunksize=CHUNKSIZE,
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
        ):
            chunk = chunk[chunk["tv_show_id"] != 0]
            if len(chunk):
                frames.append(chunk)
    sched = pd.concat(frames, ignore_index=True)
    sched["start_ts"] = pd.to_datetime(sched["start_time"], format="%d.%m.%Y %H:%M:%S").astype("int64") // 10**9
    sched["stop_ts"] = sched["start_ts"] + sched["duration"].astype("int64")
    sched = sched.sort_values(["channel_id", "start_ts"])

    index = {}
    for ch, g in sched.groupby("channel_id", sort=False):
        index[ch] = (
            g["start_ts"].to_numpy(),
            g["stop_ts"].to_numpy(),
        )
    return index, len(sched)


def main():
    sub_users = load_submission_users()
    print(f"submission users: {len(sub_users)}")

    sched_index, n_labelled_rows = build_labelled_schedule_index()
    print(f"labelled schedule rows indexed (weeks 11-30): {n_labelled_rows}")
    print(f"channels with >=1 labelled programme: {len(sched_index)}")

    users_seen = set()
    users_with_overlap = set()

    for chunk in pd.read_csv(
        VIEW_PATH,
        chunksize=CHUNKSIZE,
        usecols=["user_id", "vsetv_id", "start_time", "stop_time"],
        dtype={"user_id": "int64", "vsetv_id": "int32"},
    ):
        chunk = chunk[chunk["user_id"].isin(sub_users)]
        if chunk.empty:
            continue
        users_seen.update(chunk["user_id"].unique().tolist())

        # only bother with users not yet confirmed to have overlap
        remaining = chunk[~chunk["user_id"].isin(users_with_overlap)]
        if remaining.empty:
            continue

        vstart = pd.to_datetime(remaining["start_time"]).astype("int64").to_numpy() // 10**9
        vstop = pd.to_datetime(remaining["stop_time"]).astype("int64").to_numpy() // 10**9
        vchan = remaining["vsetv_id"].to_numpy()
        vuser = remaining["user_id"].to_numpy()

        for i in range(len(remaining)):
            ch = vchan[i]
            idx = sched_index.get(ch)
            if idx is None:
                continue
            starts, stops = idx
            vs, ve = vstart[i], vstop[i]
            # candidate programmes: start < view_stop
            j = np.searchsorted(starts, ve, side="left")
            if j == 0:
                continue
            # scan backwards a bounded window (programmes are short; a day's
            # worth of lead-in is always enough) looking for stop > view_start
            lo = max(0, j - 50)
            cand_stops = stops[lo:j]
            hit = np.any(cand_stops > vs)
            if hit:
                users_with_overlap.add(vuser[i])

    print(f"submission users present in view log: {len(users_seen)} "
          f"({100*len(users_seen)/len(sub_users):.1f}%)")
    print(f"submission users with >=1 view overlapping a labelled programme: "
          f"{len(users_with_overlap)} ({100*len(users_with_overlap)/len(sub_users):.1f}%)")


if __name__ == "__main__":
    main()
