"""
Final Hybrid v3: retuned on the mean(ALL, TOP-10, TOP-20) selection
criterion, using the LOOSE join (>=60s only) for TRAIN_FULL features
(won over strict in validation: mean=0.4804 vs 0.4783). Settings:
PPR alpha=0.85, history half_life=7d, airing-count beta=2 (TEST-window
schedule), fusion w=0.3 (history weight). No EWT/airtime-seconds boost.
Refit on TRAIN_FULL_loose, rank the TEST pool, write submission_hybrid_v3.csv.
"""
import time
import importlib.util
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR

_spec = importlib.util.spec_from_file_location("final14", DATA_DIR / "scripts" / "14_final_submissions.py")
final14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(final14)

PPR_ALPHA = 0.85
HIST_HL = 7
AIRING_BETA = 2
W = 0.3


def main():
    t0_all = time.time()
    submission_users, header_cols = final14.load_submission_template()

    train_full = pd.read_parquet(DATA_DIR / "interactions_TRAIN_FULL_loose.parquet")
    pool_test = set(pd.read_csv(DATA_DIR / "candidate_pool_TEST.csv")["tv_show_id"].tolist())
    print(f"TRAIN_FULL_loose: {len(train_full)} rows, {train_full['user_id'].nunique()} users, "
          f"{train_full['tv_show_id'].nunique()} shows")

    popularity_ranked = ce.popularity_ranking(train_full, pool_test, by="total_watched_seconds")

    t0 = time.time()
    ppr_mat, pool_items = ce.ppr_score_matrix(train_full, pool_test, submission_users, alpha=PPR_ALPHA)
    ppr_norm = ce.normalize_rows(ppr_mat)
    print(f"PPR built [{time.time()-t0:.1f}s]")

    t0 = time.time()
    hist_mat, pool_items_h = ce.history_score_matrix(
        train_full, pool_test, submission_users, basis="both", half_life=HIST_HL)
    hist_norm = ce.normalize_rows(hist_mat)
    assert pool_items == pool_items_h
    print(f"history score built [{time.time()-t0:.1f}s]")

    airing_map = ce.airing_counts(DATA_DIR / "export_arh_31-42-final.csv", pool_test)
    factor = ce.airing_factor_array(pool_items, airing_map, AIRING_BETA)
    combined = W * (hist_norm * factor[None, :]) + (1 - W) * (ppr_norm * factor[None, :])

    preds = ce.rank_scores_to_preds(combined, submission_users, pool_items,
                                     submission_users, popularity_ranked)

    fname = "submission_hybrid_v3.csv"
    final14.write_kaggle_csv(preds, submission_users, header_cols, DATA_DIR / fname)
    errs = final14.validate_kaggle_csv(DATA_DIR / fname, submission_users, header_cols)
    if errs:
        print(f"{fname}: FAILED")
        for e in errs[:10]:
            print(f"    {e}")
        raise SystemExit(1)
    print(f"{fname}: OK (1260 rows, exact user order, 5 distinct ids/row)")

    settings = f"join=loose,alpha={PPR_ALPHA},hl={HIST_HL}d,airing_beta={AIRING_BETA},w={W}"
    print(f"\nHybrid v3 settings: {settings}")
    print("validation mean(ALL,TOP10,TOP20)=0.4804 (ALL=0.6302 TOP10=0.3569 TOP20=0.4539)")
    print(f"total runtime: {time.time()-t0_all:.1f}s")

    pd.DataFrame([{"method": "Hybrid_v3", "settings": settings,
                   "map_all": 0.6302, "map_top10": 0.3569, "map_top20": 0.4539,
                   "mean_score": 0.4804, "file": fname}]
                 ).to_csv(DATA_DIR / "results_hybrid_v3_submission.csv", index=False)


if __name__ == "__main__":
    main()
