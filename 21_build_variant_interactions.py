"""
Build the extra interaction-table variants needed for the v4 one-factor
exploration:
  - count-as-distinct-viewing-days (same 120s&20% join, count redefined)
  - strict-join threshold variants for TRAINING features only:
      >=60s & >=20% of programme duration
      >=300s & >=20% of programme duration
    (VAL ground truth join stays at the original >=120s & >=20%.)
"""
import time
import importlib.util
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR

_spec = importlib.util.spec_from_file_location("build06", DATA_DIR / "scripts" / "06_build_interactions.py")
build06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build06)


def aggregate_days(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["date"] = pd.to_datetime(long_df["view_start_ts"], unit="s").dt.date
    g = long_df.groupby(["user_id", "tv_show_id"], sort=False).agg(
        count=("date", "nunique"),
        watched_seconds=("overlap_seconds", "sum"),
        last_watch_time=("view_start_ts", "max"),
    ).reset_index()
    return g


def main():
    t_all0 = time.time()

    # ---------------- count = distinct viewing days (original 120s&20% join) ----------------
    build06.MIN_OVERLAP_SEC, build06.MIN_OVERLAP_FRAC = 120, 0.20
    sched_1120 = build06.load_schedule([build06.SCHED_1120])
    sched_full = build06.load_schedule([build06.SCHED_1120, build06.SCHED_2130])

    for name, week_lo, week_hi, sched in [
        ("interactions_TRAIN_A_countdays", 11, 20, sched_1120),
        ("interactions_TRAIN_FULL_countdays", 11, 30, sched_full),
    ]:
        t1 = time.time()
        views = build06.load_views(week_lo, week_hi)
        long_df = build06.overlap_join(views, sched)
        agg = aggregate_days(long_df)
        print(f"{name}: {len(agg)} pairs, {agg['user_id'].nunique()} users, "
              f"{agg['tv_show_id'].nunique()} shows [{time.time()-t1:.1f}s]")
        build06.save(agg, name)

    # ---------------- threshold variants (keep 20% frac, vary abs-seconds floor) ----------------
    for sec, tag in [(60, "60_20"), (300, "300_20")]:
        build06.MIN_OVERLAP_SEC, build06.MIN_OVERLAP_FRAC = sec, 0.20
        print(f"\nthreshold: overlap_sec>={sec}, overlap_frac>=0.20")
        for name, week_lo, week_hi, sched in [
            (f"interactions_TRAIN_A_{tag}", 11, 20, sched_1120),
            (f"interactions_TRAIN_FULL_{tag}", 11, 30, sched_full),
        ]:
            t1 = time.time()
            views = build06.load_views(week_lo, week_hi)
            long_df = build06.overlap_join(views, sched)
            agg = build06.aggregate(long_df)
            print(f"{name}: {len(agg)} pairs, {agg['user_id'].nunique()} users, "
                  f"{agg['tv_show_id'].nunique()} shows [{time.time()-t1:.1f}s]")
            build06.save(agg, name)

    print(f"\ntotal runtime: {time.time()-t_all0:.1f}s")


if __name__ == "__main__":
    main()
