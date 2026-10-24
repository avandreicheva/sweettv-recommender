"""
One-factor-at-a-time variants around v4 (hybrid v1 with basis=count):
strict join, alpha=0.8, airing beta=1, w=0.4, half_life=14, basis=count,
PPR restart from watched_seconds. For each variant: validation MAP@5
(ALL/TOP10/TOP20), submission file + format checks. Submission/score-reading
happens separately via the kaggle CLI.
"""
import time
import importlib.util
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR

_spec = importlib.util.spec_from_file_location("final14", DATA_DIR / "scripts" / "14_final_submissions.py")
final14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(final14)

BASELINE = dict(train_table="", alpha=0.8, beta=1, w=0.4, half_life=14, basis="count",
                 restart_col="watched_seconds", fill_chain=False)

VARIANTS = {
    "hl_28":   dict(half_life=28),
    "hl_56":   dict(half_life=56),
    "hl_none": dict(half_life=None),
    "beta_1.5": dict(beta=1.5),
    "beta_2":   dict(beta=2),
    "w_0.6":    dict(w=0.6),
    "w_0.8":    dict(w=0.8),
    "ppr_restart_count": dict(restart_col="count"),
    "count_days":  dict(train_table="countdays"),
    "thresh_60_20":  dict(train_table="60_20"),
    "thresh_300_20": dict(train_table="300_20"),
}


def train_a_path(cfg):
    suf = cfg["train_table"]
    return DATA_DIR / (f"interactions_TRAIN_A.parquet" if not suf else f"interactions_TRAIN_A_{suf}.parquet")


def train_full_path(cfg):
    suf = cfg["train_table"]
    return DATA_DIR / (f"interactions_TRAIN_FULL.parquet" if not suf else f"interactions_TRAIN_FULL_{suf}.parquet")


def run_config(cfg, tag):
    t0 = time.time()
    train_a = pd.read_parquet(train_a_path(cfg))
    val = pd.read_parquet(DATA_DIR / "interactions_VAL.parquet")
    pool_set = set(pd.read_csv(DATA_DIR / "candidate_pool_VAL.csv")["tv_show_id"].tolist())
    population = set(train_a["user_id"].unique()) | set(val["user_id"].unique())
    gts = ce.build_ground_truths(val, pool_set, population)
    user_order = sorted(population)

    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")
    ppr_mat, pool_items = ce.ppr_score_matrix(train_a, pool_set, user_order, alpha=cfg["alpha"],
                                               restart_col=cfg["restart_col"])
    ppr_norm = ce.normalize_rows(ppr_mat)
    hist_mat, _ = ce.history_score_matrix(train_a, pool_set, user_order, basis=cfg["basis"], half_life=cfg["half_life"])
    hist_norm = ce.normalize_rows(hist_mat)

    airing_map = ce.airing_counts(DATA_DIR / "export_arh_21-30-final.csv", pool_set)
    factor = ce.airing_factor_array(pool_items, airing_map, cfg["beta"])
    hist_f = hist_norm * factor[None, :]
    ppr_f = ppr_norm * factor[None, :]
    combined = cfg["w"] * hist_f + (1 - cfg["w"]) * ppr_f

    if cfg["fill_chain"]:
        preds = ce.rank_with_chain([combined, ppr_f], user_order, pool_items, population, popularity_ranked)
    else:
        preds = ce.rank_scores_to_preds(combined, user_order, pool_items, population, popularity_ranked)

    res = ce.evaluate_selection(preds, gts["gt_all"], gts["gt_top10"], gts["gt_top20"])
    print(f"[{tag}] {cfg}  ALL={res['map_all']:.4f} TOP10={res['map_top10']:.4f} "
          f"TOP20={res['map_top20']:.4f} mean={res['mean_score']:.4f}  [{time.time()-t0:.1f}s]")
    return res


def build_submission(cfg, tag):
    train_full = pd.read_parquet(train_full_path(cfg))
    pool_test = set(pd.read_csv(DATA_DIR / "candidate_pool_TEST.csv")["tv_show_id"].tolist())
    submission_users, header_cols = final14.load_submission_template()

    popularity_ranked = ce.popularity_ranking(train_full, pool_test, by="total_watched_seconds")
    ppr_mat, pool_items = ce.ppr_score_matrix(train_full, pool_test, submission_users, alpha=cfg["alpha"],
                                               restart_col=cfg["restart_col"])
    ppr_norm = ce.normalize_rows(ppr_mat)
    hist_mat, _ = ce.history_score_matrix(train_full, pool_test, submission_users, basis=cfg["basis"], half_life=cfg["half_life"])
    hist_norm = ce.normalize_rows(hist_mat)

    airing_map = ce.airing_counts(DATA_DIR / "export_arh_31-42-final.csv", pool_test)
    factor = ce.airing_factor_array(pool_items, airing_map, cfg["beta"])
    hist_f = hist_norm * factor[None, :]
    ppr_f = ppr_norm * factor[None, :]
    combined = cfg["w"] * hist_f + (1 - cfg["w"]) * ppr_f

    if cfg["fill_chain"]:
        preds = ce.rank_with_chain([combined, ppr_f], submission_users, pool_items, submission_users, popularity_ranked)
    else:
        preds = ce.rank_scores_to_preds(combined, submission_users, pool_items, submission_users, popularity_ranked)

    fname = f"submission_v4_{tag}.csv"
    final14.write_kaggle_csv(preds, submission_users, header_cols, DATA_DIR / fname)
    errs = final14.validate_kaggle_csv(DATA_DIR / fname, submission_users, header_cols)
    return fname, errs


def main():
    t_all0 = time.time()
    results = []
    for tag, overrides in VARIANTS.items():
        cfg = {**BASELINE, **overrides}
        res = run_config(cfg, tag)
        fname, errs = build_submission(cfg, tag)
        status = "OK" if not errs else f"FAILED: {errs[:2]}"
        print(f"   -> {fname}: {status}")
        results.append({"variant": tag, **cfg, "map_all": res["map_all"],
                         "map_top10": res["map_top10"], "map_top20": res["map_top20"],
                         "mean_score": res["mean_score"], "file": fname, "format_ok": not errs})

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "results_v4_variants.csv", index=False)
    print("\n=== VARIANT SUMMARY (sorted by mean_score) ===")
    print(df[["variant", "map_all", "map_top10", "map_top20", "mean_score", "format_ok"]]
          .sort_values("mean_score", ascending=False).to_string(index=False))
    print(f"\ntotal runtime: {time.time()-t_all0:.1f}s")


if __name__ == "__main__":
    main()
