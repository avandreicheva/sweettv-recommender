"""
Re-tune hybrid v1 (user-history + PPR) selecting on the mean MAP@5 over
ALL / TOP-10 / TOP-20 preferred-shows truths (replacing the TOP-5-only
criterion, which Kaggle showed mis-ranks methods and rewards the harmful
EWT airtime boost). Grid: w, recency half-life, airing-count beta (incl 0),
PPR alpha -- run once per join type (strict = original >=120s&20% rule,
loose = training-features-only >=60s rule; VAL ground truth is unchanged).

Usage: python3 18_retune_hybrid_v1.py strict|loose
"""
import sys
import time
import numpy as np
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR

ALPHA_GRID = [0.5, 0.7, 0.8, 0.9]
HALF_LIFE_GRID = [None, 7, 14, 28]
BETA_GRID = [0, 0.25, 0.5, 1, 2]
W_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    join_type = sys.argv[1] if len(sys.argv) > 1 else "strict"
    assert join_type in ("strict", "loose")
    t_all0 = time.time()

    train_fname = "interactions_TRAIN_A.parquet" if join_type == "strict" else "interactions_TRAIN_A_loose.parquet"
    train_a = pd.read_parquet(DATA_DIR / train_fname)
    val = pd.read_parquet(DATA_DIR / "interactions_VAL.parquet")
    pool_set = set(pd.read_csv(DATA_DIR / "candidate_pool_VAL.csv")["tv_show_id"].tolist())
    population = set(train_a["user_id"].unique()) | set(val["user_id"].unique())
    gts = ce.build_ground_truths(val, pool_set, population)
    gt_all, gt_top10, gt_top20 = gts["gt_all"], gts["gt_top10"], gts["gt_top20"]
    user_order = sorted(population)

    print(f"[{join_type}] train rows={len(train_a)} users={train_a['user_id'].nunique()} "
          f"shows={train_a['tv_show_id'].nunique()}  population={len(population)}")

    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")
    airing_map = ce.airing_counts(DATA_DIR / "export_arh_21-30-final.csv", pool_set)

    results = []
    best = None
    n_combo = 0
    for alpha in ALPHA_GRID:
        t0 = time.time()
        ppr_mat, pool_items = ce.ppr_score_matrix(train_a, pool_set, user_order, alpha=alpha)
        ppr_norm = ce.normalize_rows(ppr_mat)
        print(f"  [alpha={alpha}] PPR built [{time.time()-t0:.1f}s]")

        for hl in HALF_LIFE_GRID:
            hist_mat, _ = ce.history_score_matrix(train_a, pool_set, user_order, basis="both", half_life=hl)
            hist_norm = ce.normalize_rows(hist_mat)

            for beta in BETA_GRID:
                factor = ce.airing_factor_array(pool_items, airing_map, beta)
                hist_f = hist_norm * factor[None, :]
                ppr_f = ppr_norm * factor[None, :]

                for w in W_GRID:
                    n_combo += 1
                    combined = w * hist_f + (1 - w) * ppr_f
                    preds = ce.rank_scores_to_preds(combined, user_order, pool_items,
                                                      population, popularity_ranked)
                    res = ce.evaluate_selection(preds, gt_all, gt_top10, gt_top20)
                    row = {"join": join_type, "alpha": alpha, "half_life": hl, "beta": beta, "w": w, **res}
                    results.append(row)
                    if best is None or res["mean_score"] > best["mean_score"]:
                        best = row

        print(f"  [alpha={alpha}] done, {n_combo} combos so far, "
              f"best so far mean_score={best['mean_score']:.4f} [{time.time()-t0:.1f}s]")

    pd.DataFrame(results).to_csv(DATA_DIR / f"results_hybrid_v1_retune_{join_type}.csv", index=False)
    print(f"\n[{join_type}] BEST: alpha={best['alpha']} half_life={best['half_life']} "
          f"beta={best['beta']} w={best['w']}  mean_score={best['mean_score']:.4f} "
          f"(ALL={best['map_all']:.4f} TOP10={best['map_top10']:.4f} TOP20={best['map_top20']:.4f})")
    print(f"total runtime: {time.time()-t_all0:.1f}s  ({n_combo} combos)")


if __name__ == "__main__":
    main()
