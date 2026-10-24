"""
Build user x tv_show_id interaction tables for three windows:
  TRAIN_A    = view-log weeks 11-20, joined to export_arh_11-20-final.csv
  VAL        = view-log weeks 21-30, joined to export_arh_21-30-final.csv
  TRAIN_FULL = view-log weeks 11-30, joined to (export_arh_11-20 + export_arh_21-30)

Join rule (per user agreement):
  - key: vsetv_id == channel_id
  - keep schedule rows with tv_show_id != 0 only (unlabelled airtime is not an item)
  - drop view rows on channels absent from the schedule entirely
  - a view counts for EVERY labelled programme whose overlap passes the
    threshold: overlap_seconds >= 120 AND overlap_seconds >= 0.2 * programme_duration
    (not just the single largest-overlap programme)

Output per window: user_id, tv_show_id, count, watched_seconds, last_watch_time
Saved as parquet (+csv) in this folder.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
VIEW_PATH = DATA_DIR / "dataset11-30.csv"
SCHED_1120 = DATA_DIR / "export_arh_11-20-final.csv"
SCHED_2130 = DATA_DIR / "export_arh_21-30-final.csv"
CHUNKSIZE = 500_000

MIN_OVERLAP_SEC = 120
MIN_OVERLAP_FRAC = 0.20
MAX_SCAN_OFFSET = 500  # safety cap; longest session (342,145s) / shortest programme (~60s) << this


def load_schedule(paths):
    frames = []
    for p in paths:
        for chunk in pd.read_csv(
            p, chunksize=CHUNKSIZE,
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
        ):
            c = chunk[chunk["tv_show_id"] != 0]
            if len(c):
                frames.append(c)
    sched = pd.concat(frames, ignore_index=True)
    sched["start_ts"] = (
        pd.to_datetime(sched["start_time"], format="%d.%m.%Y %H:%M:%S").astype("int64") // 10**9
    )
    sched["stop_ts"] = sched["start_ts"] + sched["duration"].astype("int64")
    sched = sched.drop_duplicates(subset=["channel_id", "start_ts", "tv_show_id"])
    sched = sched.sort_values(["channel_id", "start_ts"]).reset_index(drop=True)
    return sched[["channel_id", "start_ts", "stop_ts", "duration", "tv_show_id"]]


def load_views(week_lo, week_hi):
    frames = []
    for chunk in pd.read_csv(
        VIEW_PATH, chunksize=CHUNKSIZE,
        usecols=["user_id", "vsetv_id", "start_time", "stop_time"],
        dtype={"user_id": "int64", "vsetv_id": "int32"},
    ):
        start = pd.to_datetime(chunk["start_time"], format="%Y-%m-%d %H:%M:%S")
        wk = start.dt.isocalendar().week.to_numpy()
        mask = (wk >= week_lo) & (wk <= week_hi)
        if mask.any():
            sub = chunk[mask].copy()
            sub["start_ts"] = (start[mask].astype("int64") // 10**9).to_numpy()
            sub["stop_ts"] = (
                pd.to_datetime(sub["stop_time"], format="%Y-%m-%d %H:%M:%S").astype("int64") // 10**9
            )
            frames.append(sub[["user_id", "vsetv_id", "start_ts", "stop_ts"]])
    return pd.concat(frames, ignore_index=True)


def overlap_join(views: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-channel interval join. Returns long table of passing
    (user_id, tv_show_id, overlap_seconds, view_start_ts) rows."""
    out_user, out_show, out_overlap, out_vstart = [], [], [], []

    sched_by_chan = {ch: g for ch, g in sched.groupby("channel_id", sort=False)}

    for ch, vg in views.groupby("vsetv_id", sort=False):
        g = sched_by_chan.get(ch)
        if g is None:
            continue  # unscheduled channel -> drop
        p_start = g["start_ts"].to_numpy()
        p_stop = g["stop_ts"].to_numpy()
        p_dur = g["duration"].to_numpy().astype("float64")
        p_show = g["tv_show_id"].to_numpy()
        n_prog = len(p_start)

        vs = vg["start_ts"].to_numpy()
        ve = vg["stop_ts"].to_numpy()
        vu = vg["user_id"].to_numpy()

        idx0 = np.searchsorted(p_start, vs, side="right") - 1
        idx0 = np.clip(idx0, 0, n_prog - 1)

        active = np.ones(len(vs), dtype=bool)
        idx = idx0.copy()
        offset = 0
        while active.any() and offset < MAX_SCAN_OFFSET:
            cur = idx[active]
            a_vs = vs[active]
            a_ve = ve[active]
            a_vu = vu[active]

            in_range = cur < n_prog
            # stop condition for future offsets: programme start must be < view stop
            starts_before_end = np.zeros(len(cur), dtype=bool)
            starts_before_end[in_range] = p_start[cur[in_range]] < a_ve[in_range]

            do_check = in_range & starts_before_end
            if do_check.any():
                ci = cur[do_check]
                ov_start = np.maximum(a_vs[do_check], p_start[ci])
                ov_end = np.minimum(a_ve[do_check], p_stop[ci])
                ov_sec = (ov_end - ov_start).astype("float64")
                passed = (ov_sec >= MIN_OVERLAP_SEC) & (ov_sec >= MIN_OVERLAP_FRAC * p_dur[ci])
                if passed.any():
                    out_user.append(a_vu[do_check][passed])
                    out_show.append(p_show[ci][passed])
                    out_overlap.append(ov_sec[passed])
                    out_vstart.append(a_vs[do_check][passed])

            # advance only rows that are still candidates (programme started before view ended)
            next_active_local = starts_before_end
            idx_active_positions = np.flatnonzero(active)
            new_active = np.zeros(len(vs), dtype=bool)
            new_active[idx_active_positions[next_active_local]] = True
            idx[idx_active_positions[next_active_local]] += 1
            active = new_active
            offset += 1

    if not out_user:
        return pd.DataFrame(columns=["user_id", "tv_show_id", "overlap_seconds", "view_start_ts"])

    return pd.DataFrame({
        "user_id": np.concatenate(out_user),
        "tv_show_id": np.concatenate(out_show),
        "overlap_seconds": np.concatenate(out_overlap),
        "view_start_ts": np.concatenate(out_vstart),
    })


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby(["user_id", "tv_show_id"], sort=False).agg(
        count=("overlap_seconds", "size"),
        watched_seconds=("overlap_seconds", "sum"),
        last_watch_time=("view_start_ts", "max"),
    ).reset_index()
    return g


def save(df: pd.DataFrame, name: str):
    df.to_parquet(DATA_DIR / f"{name}.parquet", index=False)
    df.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"saved {name}: {len(df)} rows -> {name}.parquet / {name}.csv")


def main():
    t0 = time.time()

    print("loading schedules...")
    sched_1120 = load_schedule([SCHED_1120])
    sched_2130 = load_schedule([SCHED_2130])
    sched_full = load_schedule([SCHED_1120, SCHED_2130])
    print(f"  11-20 labelled rows: {len(sched_1120)}")
    print(f"  21-30 labelled rows: {len(sched_2130)}")
    print(f"  11-30 (combined, dedup) labelled rows: {len(sched_full)}")

    for name, week_lo, week_hi, sched in [
        ("interactions_TRAIN_A", 11, 20, sched_1120),
        ("interactions_VAL", 21, 30, sched_2130),
        ("interactions_TRAIN_FULL", 11, 30, sched_full),
    ]:
        t1 = time.time()
        views = load_views(week_lo, week_hi)
        print(f"{name}: loaded {len(views)} view rows (weeks {week_lo}-{week_hi})")
        long_df = overlap_join(views, sched)
        print(f"{name}: {len(long_df)} passing (view,programme) interaction events")
        agg = aggregate(long_df)
        print(f"{name}: {len(agg)} unique (user, tv_show_id) pairs, "
              f"{agg['user_id'].nunique()} users, {agg['tv_show_id'].nunique()} shows "
              f"[{time.time()-t1:.1f}s]")
        save(agg, name)

    print(f"\ntotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
