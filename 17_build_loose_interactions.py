"""
Looser join for TRAINING features only: any overlap >= 60s (drop the >=20%
of programme duration requirement). The VAL ground-truth join stays at the
original threshold (>=120s AND >=20%) -- only TRAIN_A and TRAIN_FULL feature
tables are rebuilt here, as *_loose variants.
"""
import time
import importlib.util
import common_eval as ce

DATA_DIR = ce.DATA_DIR

_spec = importlib.util.spec_from_file_location("build06", DATA_DIR / "scripts" / "06_build_interactions.py")
build06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build06)

# loosen the threshold for this script's calls only
build06.MIN_OVERLAP_SEC = 60
build06.MIN_OVERLAP_FRAC = 0.0


def main():
    t_all0 = time.time()
    print(f"loose join thresholds: overlap_sec>={build06.MIN_OVERLAP_SEC}, "
          f"overlap_frac>={build06.MIN_OVERLAP_FRAC}")

    sched_1120 = build06.load_schedule([build06.SCHED_1120])
    sched_full = build06.load_schedule([build06.SCHED_1120, build06.SCHED_2130])

    for name, week_lo, week_hi, sched in [
        ("interactions_TRAIN_A_loose", 11, 20, sched_1120),
        ("interactions_TRAIN_FULL_loose", 11, 30, sched_full),
    ]:
        t1 = time.time()
        views = build06.load_views(week_lo, week_hi)
        long_df = build06.overlap_join(views, sched)
        agg = build06.aggregate(long_df)
        print(f"{name}: {len(agg)} unique (user, tv_show_id) pairs, "
              f"{agg['user_id'].nunique()} users, {agg['tv_show_id'].nunique()} shows "
              f"[{time.time()-t1:.1f}s]")
        build06.save(agg, name)

    print(f"\ntotal runtime: {time.time()-t_all0:.1f}s")


if __name__ == "__main__":
    main()
