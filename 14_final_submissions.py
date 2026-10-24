"""
Refit the validated methods (popularity, user-history, ALS, PPR, best hybrid)
on TRAIN_FULL (weeks 11-30), rank only the TEST candidate pool (31-42
schedule), and write one Kaggle submission file per method -- same header,
same user order, 1,260 rows, 5 distinct space-separated tv_show_id per row.
"""
import time
import importlib.util
import numpy as np
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR
K = ce.K
SEED = ce.SEED

_spec = importlib.util.spec_from_file_location("als", DATA_DIR / "scripts" / "11_als.py")
als_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(als_mod)

# validated settings
HIST_BASIS, HIST_HL = "both", 14
PPR_ALPHA = 0.8
ALS_FACTORS, ALS_A = 20, 0.05
HYBRID_W, HYBRID_BETA = 0.4, 1


def load_submission_template():
    sub = pd.read_csv(DATA_DIR / "submission.csv")
    return sub["user_id"].tolist(), sub.columns.tolist()


def write_kaggle_csv(preds: dict, user_order: list, header_cols: list, path):
    rows = []
    for u in user_order:
        ids = preds[u]
        assert len(ids) == K, f"user {u} has {len(ids)} ids, expected {K}"
        assert len(set(ids)) == K, f"user {u} has duplicate ids: {ids}"
        rows.append({header_cols[0]: u, header_cols[1]: " ".join(str(int(x)) for x in ids)})
    pd.DataFrame(rows, columns=header_cols).to_csv(path, index=False)


def validate_kaggle_csv(path, expected_users, header_cols):
    df = pd.read_csv(path, dtype={header_cols[0]: "int64"})
    errors = []
    if list(df.columns) != header_cols:
        errors.append(f"header mismatch: {list(df.columns)} != {header_cols}")
    if len(df) != len(expected_users):
        errors.append(f"row count {len(df)} != {len(expected_users)}")
    if df[header_cols[0]].tolist() != expected_users:
        errors.append("user_id order does not match submission.csv template")
    for i, val in enumerate(df[header_cols[1]]):
        ids = val.split(" ")
        if len(ids) != K:
            errors.append(f"row {i} user {df[header_cols[0]].iloc[i]}: {len(ids)} ids, expected {K}")
            continue
        try:
            ids_int = [int(x) for x in ids]
        except ValueError:
            errors.append(f"row {i}: non-integer id(s) in '{val}'")
            continue
        if len(set(ids_int)) != K:
            errors.append(f"row {i} user {df[header_cols[0]].iloc[i]}: duplicate ids {ids_int}")
    return errors


def main():
    t_all0 = time.time()
    submission_users, header_cols = load_submission_template()
    print(f"submission template: {len(submission_users)} users, header={header_cols}")

    train_full = pd.read_parquet(DATA_DIR / "interactions_TRAIN_FULL.parquet")
    pool_test = set(pd.read_csv(DATA_DIR / "candidate_pool_TEST.csv")["tv_show_id"].tolist())
    print(f"TRAIN_FULL: {len(train_full)} rows, {train_full['user_id'].nunique()} users, "
          f"{train_full['tv_show_id'].nunique()} shows")
    print(f"TEST candidate pool: {len(pool_test)} shows")

    results = []

    # ---------------- Popularity ----------------
    t0 = time.time()
    popularity_ranked = ce.popularity_ranking(train_full, pool_test, by="total_watched_seconds")
    top5 = popularity_ranked[:K].tolist()
    preds_pop = {u: top5 for u in submission_users}
    fname = "submission_popularity.csv"
    write_kaggle_csv(preds_pop, submission_users, header_cols, DATA_DIR / fname)
    results.append({"method": "Popularity", "settings": "by=total_watched_seconds",
                     "map_all": 0.5420, "map_top5": 0.1521, "file": fname,
                     "time_s": time.time() - t0})
    print(f"Popularity -> {fname} [{time.time()-t0:.1f}s]")

    # ---------------- User-history ----------------
    t0 = time.time()
    hist_mat, pool_items = ce.history_score_matrix(
        train_full, pool_test, submission_users, basis=HIST_BASIS, half_life=HIST_HL)
    preds_hist = ce.rank_scores_to_preds(hist_mat, submission_users, pool_items,
                                          submission_users, popularity_ranked)
    fname = "submission_user_history.csv"
    write_kaggle_csv(preds_hist, submission_users, header_cols, DATA_DIR / fname)
    results.append({"method": "User-history", "settings": f"basis={HIST_BASIS},half_life={HIST_HL}d",
                     "map_all": 0.5894, "map_top5": 0.2455, "file": fname,
                     "time_s": time.time() - t0})
    print(f"User-history -> {fname} [{time.time()-t0:.1f}s]")

    # ---------------- ALS ----------------
    t0 = time.time()
    user_ids = sorted(train_full["user_id"].unique())
    item_ids = sorted(train_full["tv_show_id"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {s: i for i, s in enumerate(item_ids)}
    pool_items_in_vocab = [s for s in item_ids if s in pool_test]
    pool_cols = np.array([item_index[s] for s in pool_items_in_vocab])
    pool_show_ids = np.array(pool_items_in_vocab)

    conf_csr = als_mod.build_conf_csr(train_full, user_index, item_index, ALS_A)
    U, V = als_mod.als_fit(conf_csr, n_factors=ALS_FACTORS, seed=SEED)
    scores = U @ V[pool_cols].T

    preds_als = {}
    for u in submission_users:
        uidx = user_index.get(u)
        if uidx is None:
            preds_als[u] = popularity_ranked[:K].tolist()
            continue
        row = scores[uidx]
        top_local = np.argpartition(-row, min(K, len(row) - 1))[:K]
        top_local = top_local[np.argsort(-row[top_local])]
        rec = pool_show_ids[top_local].tolist()
        preds_als[u] = ce.fill_with_popularity(rec, popularity_ranked, k=K)
    fname = "submission_als.csv"
    write_kaggle_csv(preds_als, submission_users, header_cols, DATA_DIR / fname)
    results.append({"method": "ALS", "settings": f"factors={ALS_FACTORS},a={ALS_A}",
                     "map_all": 0.5315, "map_top5": 0.1412, "file": fname,
                     "time_s": time.time() - t0})
    print(f"ALS -> {fname} [{time.time()-t0:.1f}s]")

    # ---------------- PPR ----------------
    t0 = time.time()
    ppr_mat, pool_items_p = ce.ppr_score_matrix(train_full, pool_test, submission_users, alpha=PPR_ALPHA)
    preds_ppr = ce.rank_scores_to_preds(ppr_mat, submission_users, pool_items_p,
                                         submission_users, popularity_ranked)
    fname = "submission_ppr.csv"
    write_kaggle_csv(preds_ppr, submission_users, header_cols, DATA_DIR / fname)
    results.append({"method": "PPR", "settings": f"alpha={PPR_ALPHA}",
                     "map_all": 0.5806, "map_top5": 0.2395, "file": fname,
                     "time_s": time.time() - t0})
    print(f"PPR -> {fname} [{time.time()-t0:.1f}s]")

    # ---------------- Hybrid (best: w=0.4 fusion + airing factor beta=1) ----------------
    t0 = time.time()
    hist_norm = ce.normalize_rows(hist_mat)
    ppr_norm = ce.normalize_rows(ppr_mat)
    assert pool_items == pool_items_p
    combined = HYBRID_W * hist_norm + (1 - HYBRID_W) * ppr_norm
    airing_map = ce.airing_counts(DATA_DIR / "export_arh_31-42-final.csv", pool_test)
    factor = ce.airing_factor_array(pool_items, airing_map, HYBRID_BETA)
    hybrid_scored = combined * factor[None, :]
    preds_hybrid = ce.rank_scores_to_preds(hybrid_scored, submission_users, pool_items,
                                            submission_users, popularity_ranked)
    fname = "submission_hybrid.csv"
    write_kaggle_csv(preds_hybrid, submission_users, header_cols, DATA_DIR / fname)
    results.append({"method": "Hybrid", "settings": f"w={HYBRID_W},airing_beta={HYBRID_BETA}",
                     "map_all": 0.6222, "map_top5": 0.2504, "file": fname,
                     "time_s": time.time() - t0})
    print(f"Hybrid -> {fname} [{time.time()-t0:.1f}s]")

    # ---------------- validate all files ----------------
    print("\n=== validation ===")
    any_errors = False
    for r in results:
        errs = validate_kaggle_csv(DATA_DIR / r["file"], submission_users, header_cols)
        if errs:
            any_errors = True
            print(f"{r['file']}: FAILED")
            for e in errs[:5]:
                print(f"    {e}")
        else:
            print(f"{r['file']}: OK (1260 rows, exact user order, 5 distinct ids/row)")

    print("\n=== FINAL TABLE ===")
    df = pd.DataFrame(results)[["method", "settings", "map_all", "map_top5", "file", "time_s"]]
    print(df.to_string(index=False))
    df.to_csv(DATA_DIR / "final_submission_table.csv", index=False)

    print(f"\ntotal runtime: {time.time()-t_all0:.1f}s  (seed={SEED})")
    if any_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
