"""
Refit Hybrid v2 (expected-watch-time + PPR) on TRAIN_FULL, rank the TEST
pool, write submission_hybrid_v2.csv with the same format checks as the
other submission files.
"""
import time
import importlib.util
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR
SEED = ce.SEED

_spec = importlib.util.spec_from_file_location("final14", DATA_DIR / "scripts" / "14_final_submissions.py")
final14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(final14)

PPR_ALPHA = 0.8
EWT_MODE, EWT_HL, EWT_BETA = "seconds", 14, 0.5
HYBRID_V2_W = 0.2

TRAIN_SCHEDULE_FULL = [DATA_DIR / "export_arh_11-20-final.csv", DATA_DIR / "export_arh_21-30-final.csv"]
TARGET_SCHEDULE_TEST = [DATA_DIR / "export_arh_31-42-final.csv"]


def main():
    t0_all = time.time()
    submission_users, header_cols = final14.load_submission_template()

    train_full = pd.read_parquet(DATA_DIR / "interactions_TRAIN_FULL.parquet")
    pool_test = set(pd.read_csv(DATA_DIR / "candidate_pool_TEST.csv")["tv_show_id"].tolist())
    popularity_ranked = ce.popularity_ranking(train_full, pool_test, by="total_watched_seconds")

    t0 = time.time()
    ppr_mat, pool_items = ce.ppr_score_matrix(train_full, pool_test, submission_users, alpha=PPR_ALPHA)
    ppr_norm = ce.normalize_rows(ppr_mat)
    print(f"PPR matrix (TRAIN_FULL) built [{time.time()-t0:.1f}s]")

    t0 = time.time()
    ewt_mat, pool_items_e = ce.expected_watch_time_matrix(
        train_full, pool_test, submission_users, TRAIN_SCHEDULE_FULL, TARGET_SCHEDULE_TEST,
        engagement_mode=EWT_MODE, half_life=EWT_HL, beta=EWT_BETA)
    assert pool_items == pool_items_e
    ewt_norm = ce.normalize_rows(ewt_mat)
    print(f"EWT matrix (TRAIN_FULL) built [{time.time()-t0:.1f}s]")

    fused = HYBRID_V2_W * ewt_norm + (1 - HYBRID_V2_W) * ppr_norm
    preds = ce.rank_with_chain([fused, ppr_norm], submission_users, pool_items,
                               submission_users, popularity_ranked)

    fname = "submission_hybrid_v2.csv"
    final14.write_kaggle_csv(preds, submission_users, header_cols, DATA_DIR / fname)
    print(f"Hybrid v2 -> {fname}")

    errs = final14.validate_kaggle_csv(DATA_DIR / fname, submission_users, header_cols)
    if errs:
        print(f"{fname}: FAILED")
        for e in errs[:10]:
            print(f"    {e}")
        raise SystemExit(1)
    print(f"{fname}: OK (1260 rows, exact user order, 5 distinct ids/row)")

    settings = f"ewt_mode={EWT_MODE},ewt_half_life={EWT_HL}d,ewt_beta={EWT_BETA},ppr_alpha={PPR_ALPHA},w={HYBRID_V2_W}"
    row = {"method": "Hybrid_v2", "settings": settings, "map_all": 0.6042, "map_top5": 0.2522,
           "file": fname, "time_s": time.time() - t0_all}
    print("\n=== Hybrid v1 vs Hybrid v2 ===")
    print(f"Hybrid v1: settings=w=0.4,airing_beta=1            MAP@5(ALL)=0.6222 MAP@5(TOP5)=0.2504 file=submission_hybrid.csv")
    print(f"Hybrid v2: settings={settings} MAP@5(ALL)={row['map_all']:.4f} MAP@5(TOP5)={row['map_top5']:.4f} file={fname}")

    pd.DataFrame([row]).to_csv(DATA_DIR / "results_hybrid_v2_submission.csv", index=False)
    print(f"\ntotal runtime: {time.time()-t0_all:.1f}s  (seed={SEED})")


if __name__ == "__main__":
    main()
