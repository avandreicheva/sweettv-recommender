"""
Expected-watch-time ranker: score(u,i) = engagement(u,i) * airtime(i, target
window)^beta, tuned on the TOP-5 preferred validation truth.
  engagement = watched_seconds(u,i) [/ show's total scheduled airtime in the
  user's TRAINING window, if mode='ratio'] * recency decay
  airtime    = total scheduled seconds of show i in the TARGET-window
  SCHEDULE (export_arh_21-30-final.csv for VAL tuning)

Missing-slot fallback chain: EWT ranker's own picks -> PPR picks -> popularity.

Then re-tunes the hybrid with this ranker in place of user-history
(fusion weight w, same fallback chain [fused, PPR] -> popularity), and
compares against the current hybrid (0.6222 / 0.2504).
"""
import time
import numpy as np
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR
PPR_ALPHA = 0.8
ENGAGEMENT_MODES = ["ratio", "seconds"]
HALF_LIFE_GRID = [None, 28, 14]
BETA_GRID = [0, 0.5, 1, 1.5]
W_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

TRAIN_SCHEDULE = [DATA_DIR / "export_arh_11-20-final.csv"]
TARGET_SCHEDULE_VAL = [DATA_DIR / "export_arh_21-30-final.csv"]

CURRENT_HYBRID = {"map_all": 0.6222, "map_top5": 0.2504}


def main():
    t_all0 = time.time()
    data = ce.load_core()
    train_a, pool_set = data["train_a"], data["pool_set"]
    population, gt_all, gt_top5 = data["population"], data["gt_all"], data["gt_top5"]
    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")
    user_order = sorted(population)

    t0 = time.time()
    ppr_mat, pool_items = ce.ppr_score_matrix(train_a, pool_set, user_order, alpha=PPR_ALPHA)
    ppr_norm = ce.normalize_rows(ppr_mat)
    print(f"PPR score matrix built [{time.time()-t0:.1f}s]")

    def evaluate_chain(primary_mat):
        preds = ce.rank_with_chain([primary_mat, ppr_norm], user_order, pool_items,
                                    population, popularity_ranked)
        return ce.evaluate_both(preds, gt_all, gt_top5)

    print("\n=== Stage 1: tune expected-watch-time ranker "
          "(engagement_mode x half_life x beta), fallback=[EWT, PPR, popularity] ===")
    results = []
    best_ewt = None
    for mode in ENGAGEMENT_MODES:
        for hl in HALF_LIFE_GRID:
            for beta in BETA_GRID:
                t0 = time.time()
                mat, pool_items_e = ce.expected_watch_time_matrix(
                    train_a, pool_set, user_order, TRAIN_SCHEDULE, TARGET_SCHEDULE_VAL,
                    engagement_mode=mode, half_life=hl, beta=beta)
                assert pool_items_e == pool_items
                mat_norm = ce.normalize_rows(mat)
                res_all, res_top5 = evaluate_chain(mat_norm)
                dt = time.time() - t0
                hl_str = "none" if hl is None else f"{hl}d"
                print(f"  mode={mode:7s} hl={hl_str:5s} beta={beta:3} "
                      f"MAP@5(ALL)={res_all['map@k']:.4f} MAP@5(TOP5)={res_top5['map@k']:.4f} [{dt:.2f}s]")
                row = {"mode": mode, "half_life": hl_str, "beta": beta,
                       "map_all": res_all["map@k"], "map_top5": res_top5["map@k"], "time_s": dt}
                results.append(row)
                if best_ewt is None or row["map_top5"] > best_ewt["map_top5"]:
                    best_ewt = row

    print(f"\n-> best EWT config: mode={best_ewt['mode']} half_life={best_ewt['half_life']} "
          f"beta={best_ewt['beta']}  MAP@5(ALL)={best_ewt['map_all']:.4f} "
          f"MAP@5(TOP5)={best_ewt['map_top5']:.4f}")
    pd.DataFrame(results).to_csv(DATA_DIR / "results_ewt_grid.csv", index=False)

    best_hl = None if best_ewt["half_life"] == "none" else int(best_ewt["half_life"][:-1])
    ewt_mat, _ = ce.expected_watch_time_matrix(
        train_a, pool_set, user_order, TRAIN_SCHEDULE, TARGET_SCHEDULE_VAL,
        engagement_mode=best_ewt["mode"], half_life=best_hl, beta=best_ewt["beta"])
    ewt_norm = ce.normalize_rows(ewt_mat)

    print("\n=== Stage 2: re-tune hybrid fusion weight w (EWT + PPR) ===")
    best_hyb = None
    hyb_results = []
    for w in W_GRID:
        t0 = time.time()
        fused = w * ewt_norm + (1 - w) * ppr_norm
        preds = ce.rank_with_chain([fused, ppr_norm], user_order, pool_items,
                                    population, popularity_ranked)
        res_all, res_top5 = ce.evaluate_both(preds, gt_all, gt_top5)
        dt = time.time() - t0
        print(f"  w={w:.1f}  MAP@5(ALL)={res_all['map@k']:.4f} MAP@5(TOP5)={res_top5['map@k']:.4f} [{dt:.2f}s]")
        row = {"w": w, "map_all": res_all["map@k"], "map_top5": res_top5["map@k"], "time_s": dt}
        hyb_results.append(row)
        if best_hyb is None or row["map_top5"] > best_hyb["map_top5"]:
            best_hyb = row

    print(f"\n-> best Hybrid v2 config: w={best_hyb['w']}  "
          f"MAP@5(ALL)={best_hyb['map_all']:.4f} MAP@5(TOP5)={best_hyb['map_top5']:.4f}")
    pd.DataFrame(hyb_results).to_csv(DATA_DIR / "results_hybrid_v2_grid.csv", index=False)

    print("\n=== Hybrid v1 (user-history+PPR) vs Hybrid v2 (expected-watch-time+PPR) ===")
    print(f"Hybrid v1: MAP@5(ALL)={CURRENT_HYBRID['map_all']:.4f} "
          f"MAP@5(TOP5)={CURRENT_HYBRID['map_top5']:.4f}")
    print(f"Hybrid v2: MAP@5(ALL)={best_hyb['map_all']:.4f} "
          f"MAP@5(TOP5)={best_hyb['map_top5']:.4f}  "
          f"(Δ TOP5 = {best_hyb['map_top5']-CURRENT_HYBRID['map_top5']:+.4f})")

    pd.DataFrame([{
        "method": "Hybrid_v2[EWT+PPR]",
        "ewt_mode": best_ewt["mode"], "ewt_half_life": best_ewt["half_life"],
        "ewt_beta": best_ewt["beta"], "w": best_hyb["w"],
        "map_all": best_hyb["map_all"], "map_top5": best_hyb["map_top5"],
    }]).to_csv(DATA_DIR / "results_hybrid_v2_best.csv", index=False)

    print(f"\ntotal runtime: {time.time()-t_all0:.1f}s")


if __name__ == "__main__":
    main()
