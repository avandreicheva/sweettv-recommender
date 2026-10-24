"""
Popularity baseline + user-history ranker, evaluated on VAL (weeks 21-30)
with MAP@5. Fit window = TRAIN_A (weeks 11-20). Candidate pool = distinct
labelled shows airing in the export_arh_21-30-final.csv schedule.

Population scored = union of users seen in TRAIN_A and users seen in VAL
(so cold-start users with no TRAIN_A history and no-overlap users with
empty R_u are both represented; empty R_u is excluded from the mean but
counted per the evaluator contract).
"""
import time
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
SEED = 42
K = 5
RECENCY_HALF_LIFE_DAYS = 14

spec = importlib.util.spec_from_file_location("evaluator", DATA_DIR / "scripts" / "08_evaluator.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    train_a = pd.read_parquet(DATA_DIR / "interactions_TRAIN_A.parquet")
    val = pd.read_parquet(DATA_DIR / "interactions_VAL.parquet")
    pool = pd.read_csv(DATA_DIR / "candidate_pool_VAL.csv")["tv_show_id"]
    pool_set = set(pool.tolist())
    print(f"TRAIN_A: {len(train_a)} rows, {train_a['user_id'].nunique()} users")
    print(f"VAL: {len(val)} rows, {val['user_id'].nunique()} users")
    print(f"candidate pool (VAL window): {len(pool_set)} shows")

    # ---- ground truth R_u and population ----
    val_pool = val[val["tv_show_id"].isin(pool_set)]
    ground_truth = val_pool.groupby("user_id")["tv_show_id"].apply(set).to_dict()

    population = set(train_a["user_id"].unique()) | set(val["user_id"].unique())
    ground_truth_full = {u: ground_truth.get(u, set()) for u in population}

    ru_sizes = np.array([len(s) for s in ground_truth_full.values()])
    print(f"\npopulation (TRAIN_A users U VAL users): {len(population)}")
    print("R_u size stats: "
          f"mean={ru_sizes.mean():.2f} median={np.median(ru_sizes):.0f} "
          f"min={ru_sizes.min()} max={ru_sizes.max()}")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  p{int(q*100)} = {np.quantile(ru_sizes, q):.0f}")
    n_empty = int((ru_sizes == 0).sum())
    print(f"users with empty R_u: {n_empty} ({100*n_empty/len(population):.1f}%)")

    # ---- popularity baseline (fit on TRAIN_A, restricted to pool) ----
    t0 = time.time()
    train_a_pool = train_a[train_a["tv_show_id"].isin(pool_set)]
    pop_counts = train_a_pool.groupby("tv_show_id")["user_id"].nunique()
    pop_counts = pop_counts.reindex(sorted(pool_set), fill_value=0)
    jitter = rng.random(len(pop_counts)) * 1e-6
    pop_rank = (pop_counts.to_numpy() + jitter).argsort()[::-1]
    popularity_ranked = pop_counts.index.to_numpy()[pop_rank]
    popularity_top5 = popularity_ranked[:K].tolist()

    preds_popularity = {u: popularity_top5 for u in population}
    t_pop_build = time.time() - t0

    # ---- user-history ranker ----
    t0 = time.time()
    hist = train_a[train_a["tv_show_id"].isin(pool_set)].copy()
    anchor_t = train_a["last_watch_time"].max()  # "now" reference for recency decay
    hist["days_since"] = (anchor_t - hist["last_watch_time"]) / 86400.0
    hist["score"] = hist["watched_seconds"] * np.power(0.5, hist["days_since"] / RECENCY_HALF_LIFE_DAYS)
    hist["jitter"] = rng.random(len(hist)) * 1e-6
    hist["score_tiebrk"] = hist["score"] + hist["jitter"]

    hist_sorted = hist.sort_values(["user_id", "score_tiebrk"], ascending=[True, False])
    user_hist_lists = hist_sorted.groupby("user_id")["tv_show_id"].apply(list).to_dict()

    preds_history = {}
    for u in population:
        own = user_hist_lists.get(u, [])
        rec = list(own[:K])
        if len(rec) < K:
            seen = set(rec)
            for show in popularity_ranked:
                if show not in seen:
                    rec.append(show)
                    seen.add(show)
                    if len(rec) == K:
                        break
        preds_history[u] = rec
    t_hist_build = time.time() - t0

    # ---- evaluate ----
    t0 = time.time()
    res_pop = evaluator.map_at_k(preds_popularity, ground_truth_full, k=K)
    t_pop_eval = time.time() - t0

    t0 = time.time()
    res_hist = evaluator.map_at_k(preds_history, ground_truth_full, k=K)
    t_hist_eval = time.time() - t0

    print(f"\n=== Popularity baseline ===")
    print(f"top5 (global): {popularity_top5}")
    print(f"MAP@5 = {res_pop['map@k']:.4f}  "
          f"(scored {res_pop['n_scored']}/{res_pop['n_total']}, "
          f"empty R_u excluded: {res_pop['n_empty_Ru']})")
    print(f"build time: {t_pop_build:.2f}s, eval time: {t_pop_eval:.2f}s")

    print(f"\n=== User-history ranker (recency-weighted watched_seconds, "
          f"half-life={RECENCY_HALF_LIFE_DAYS}d, popularity fill) ===")
    print(f"MAP@5 = {res_hist['map@k']:.4f}  "
          f"(scored {res_hist['n_scored']}/{res_hist['n_total']}, "
          f"empty R_u excluded: {res_hist['n_empty_Ru']})")
    print(f"build time: {t_hist_build:.2f}s, eval time: {t_hist_eval:.2f}s")

    # ---- save prediction artifacts ----
    def to_df(preds):
        rows = [{"user_id": u, "tv_show_id": " ".join(map(str, p))} for u, p in preds.items()]
        return pd.DataFrame(rows)

    to_df(preds_popularity).to_csv(DATA_DIR / "preds_popularity_VAL.csv", index=False)
    to_df(preds_history).to_csv(DATA_DIR / "preds_user_history_VAL.csv", index=False)
    pd.DataFrame({"user_id": list(ground_truth_full.keys()),
                  "tv_show_id": [" ".join(map(str, sorted(s))) for s in ground_truth_full.values()]
                  }).to_csv(DATA_DIR / "ground_truth_VAL.csv", index=False)

    print(f"\nsaved preds_popularity_VAL.csv, preds_user_history_VAL.csv, ground_truth_VAL.csv")
    print(f"total runtime: {time.time()-t_start:.1f}s  (seed={SEED})")


if __name__ == "__main__":
    main()
