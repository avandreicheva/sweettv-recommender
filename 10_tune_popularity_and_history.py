"""
Tune (on validation):
  - Popularity: rank by distinct_viewers vs total_watched_seconds
  - User-history: score basis in {seconds, count, both} x recency half-life
    in {7, 14, 28, None} days, popularity-filled for missing slots.
Selection criterion: best MAP@5 on the TOP5 (preferred-shows) ground truth,
since that is what the competition actually targets; ALL is reported
alongside for diagnostic purposes.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
import common_eval as ce

DATA_DIR = ce.DATA_DIR


def main():
    t_all0 = time.time()
    data = ce.load_core()
    train_a, pool_set = data["train_a"], data["pool_set"]
    population, gt_all, gt_top5 = data["population"], data["gt_all"], data["gt_top5"]

    print(f"population={len(population)}  pool={len(pool_set)}")
    print(f"gt_all empty_Ru share handled by evaluator; "
          f"gt_top5 sizes -> mean={np.mean([len(s) for s in gt_top5.values()]):.2f}")

    # ---------------- Popularity: tune ranking signal ----------------
    print("\n=== Popularity: tuning ranking signal ===")
    best_pop = None
    for by in ["distinct_viewers", "total_watched_seconds"]:
        t0 = time.time()
        ranked = ce.popularity_ranking(train_a, pool_set, by=by)
        top5 = ranked[: ce.K].tolist()
        preds = {u: top5 for u in population}
        res_all, res_top5 = ce.evaluate_both(preds, gt_all, gt_top5)
        dt = time.time() - t0
        print(f"  by={by:22s} MAP@5(ALL)={res_all['map@k']:.4f} "
              f"MAP@5(TOP5)={res_top5['map@k']:.4f}  top5={top5}  [{dt:.2f}s]")
        if best_pop is None or res_top5["map@k"] > best_pop["top5"]:
            best_pop = {"by": by, "all": res_all["map@k"], "top5": res_top5["map@k"],
                        "ranked": ranked, "time": dt}

    print(f"  -> best popularity variant: by={best_pop['by']}")

    # ---------------- User-history: grid over basis x half-life ----------------
    print("\n=== User-history: tuning score basis x recency half-life ===")
    basis_opts = ["seconds", "count", "both"]
    halflife_opts = [7, 14, 28, None]

    best_hist = None
    for basis in basis_opts:
        for hl in halflife_opts:
            t0 = time.time()
            hist = train_a[train_a["tv_show_id"].isin(pool_set)].copy()
            anchor_t = train_a["last_watch_time"].max()
            if hl is None:
                decay = 1.0
            else:
                days_since = (anchor_t - hist["last_watch_time"]) / 86400.0
                decay = np.power(0.5, days_since / hl)

            if basis == "seconds":
                raw = hist["watched_seconds"].to_numpy()
            elif basis == "count":
                raw = hist["count"].to_numpy()
            else:  # both
                raw = hist["watched_seconds"].to_numpy() * hist["count"].to_numpy()

            rng = np.random.default_rng(ce.SEED)
            jitter = rng.random(len(hist)) * 1e-6
            hist["score"] = raw * decay + jitter

            hist_sorted = hist.sort_values(["user_id", "score"], ascending=[True, False])
            user_hist_lists = hist_sorted.groupby("user_id")["tv_show_id"].apply(list).to_dict()

            preds = {}
            for u in population:
                own = user_hist_lists.get(u, [])[: ce.K]
                preds[u] = ce.fill_with_popularity(own, best_pop["ranked"], k=ce.K)

            res_all, res_top5 = ce.evaluate_both(preds, gt_all, gt_top5)
            dt = time.time() - t0
            hl_str = "none" if hl is None else f"{hl}d"
            print(f"  basis={basis:7s} half_life={hl_str:5s} "
                  f"MAP@5(ALL)={res_all['map@k']:.4f} MAP@5(TOP5)={res_top5['map@k']:.4f} [{dt:.2f}s]")

            if best_hist is None or res_top5["map@k"] > best_hist["top5"]:
                best_hist = {"basis": basis, "half_life": hl_str,
                             "all": res_all["map@k"], "top5": res_top5["map@k"], "time": dt}

    print(f"\n-> best user-history config: basis={best_hist['basis']} half_life={best_hist['half_life']}")

    print("\n=== SUMMARY (best per method) ===")
    print(f"Popularity[{best_pop['by']}]: MAP@5(ALL)={best_pop['all']:.4f} "
          f"MAP@5(TOP5)={best_pop['top5']:.4f}")
    print(f"User-history[basis={best_hist['basis']},hl={best_hist['half_life']}]: "
          f"MAP@5(ALL)={best_hist['all']:.4f} MAP@5(TOP5)={best_hist['top5']:.4f}")
    print(f"\ntotal runtime: {time.time()-t_all0:.1f}s")

    # persist chosen configs + results for the final compact table
    pd.DataFrame([
        {"method": f"Popularity[{best_pop['by']}]", "map_all": best_pop["all"],
         "map_top5": best_pop["top5"], "time_s": best_pop["time"]},
        {"method": f"UserHistory[basis={best_hist['basis']},hl={best_hist['half_life']}]",
         "map_all": best_hist["all"], "map_top5": best_hist["top5"], "time_s": best_hist["time"]},
    ]).to_csv(DATA_DIR / "results_popularity_history.csv", index=False)


if __name__ == "__main__":
    main()
