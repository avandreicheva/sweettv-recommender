"""
Hybrid = score fusion of normalized user-history and PPR scores, tuned on
the TOP-5 (preferred-shows) validation truth. Also tests an airing-frequency
factor (1+log(airings_in_target_window_schedule))^beta applied to history,
PPR, and the hybrid separately, reporting each component's gain.

airing_factor is linear in the fused score (factor(w*h + (1-w)*p) =
w*factor*h + (1-w)*factor*p), so "hybrid + airing" is computed as a single
post-fusion multiply -- mathematically identical to applying it to both
components before fusing.
"""
import time
import numpy as np
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR
BEST_HIST_BASIS, BEST_HIST_HL = "both", 14
BEST_PPR_ALPHA = 0.8
W_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
BETA_GRID = [0, 0.25, 0.5, 1, 2]


def main():
    t0_all = time.time()
    data = ce.load_core()
    train_a, pool_set = data["train_a"], data["pool_set"]
    population, gt_all, gt_top5 = data["population"], data["gt_all"], data["gt_top5"]
    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")
    user_order = sorted(population)

    t0 = time.time()
    hist_mat, pool_items = ce.history_score_matrix(
        train_a, pool_set, user_order, basis=BEST_HIST_BASIS, half_life=BEST_HIST_HL)
    hist_norm = ce.normalize_rows(hist_mat)
    print(f"history score matrix built [{time.time()-t0:.1f}s]")

    t0 = time.time()
    ppr_mat, pool_items_p = ce.ppr_score_matrix(train_a, pool_set, user_order, alpha=BEST_PPR_ALPHA)
    ppr_norm = ce.normalize_rows(ppr_mat)
    assert pool_items == pool_items_p
    print(f"PPR score matrix built [{time.time()-t0:.1f}s]")

    def run(mat, label):
        t0 = time.time()
        preds = ce.rank_scores_to_preds(mat, user_order, pool_items, population, popularity_ranked)
        res_all, res_top5 = ce.evaluate_both(preds, gt_all, gt_top5)
        dt = time.time() - t0
        print(f"  {label:45s} MAP@5(ALL)={res_all['map@k']:.4f} "
              f"MAP@5(TOP5)={res_top5['map@k']:.4f}  [{dt:.2f}s]")
        return res_all["map@k"], res_top5["map@k"], dt

    results = []

    # ---------------- Stage 1: tune fusion weight w (no airing factor) ----------------
    print("\n=== Stage 1: fusion weight w (hybrid, no airing factor) ===")
    best_w, best_w_top5, best_w_all, best_w_time = None, -1, None, None
    for w in W_GRID:
        combined = w * hist_norm + (1 - w) * ppr_norm
        a, t5, dt = run(combined, f"w={w:.1f}")
        results.append({"stage": "fusion_weight", "w": w, "beta": None,
                         "map_all": a, "map_top5": t5, "time_s": dt})
        if t5 > best_w_top5:
            best_w, best_w_top5, best_w_all, best_w_time = w, t5, a, dt
    print(f"-> best w={best_w}  MAP@5(TOP5)={best_w_top5:.4f}")
    hybrid_base = best_w * hist_norm + (1 - best_w) * ppr_norm

    # reference: pure history (w=1) and pure PPR (w=0) already in the grid above
    hist_alone_all, hist_alone_top5 = [r for r in results if r["w"] == 1.0][0]["map_all"], \
        [r for r in results if r["w"] == 1.0][0]["map_top5"]
    ppr_alone_all, ppr_alone_top5 = [r for r in results if r["w"] == 0.0][0]["map_all"], \
        [r for r in results if r["w"] == 0.0][0]["map_top5"]

    # ---------------- airing-frequency factor ----------------
    airing_map = ce.airing_counts(DATA_DIR / "export_arh_21-30-final.csv", pool_set)

    def tune_beta(mat, label):
        best = None
        for beta in BETA_GRID:
            factor = ce.airing_factor_array(pool_items, airing_map, beta)
            scored = mat * factor[None, :]
            a, t5, dt = run(scored, f"{label} beta={beta}")
            results.append({"stage": label, "w": None, "beta": beta,
                             "map_all": a, "map_top5": t5, "time_s": dt})
            if best is None or t5 > best["top5"]:
                best = {"beta": beta, "all": a, "top5": t5, "time": dt}
        return best

    print("\n=== Stage 2: airing-frequency factor on HISTORY alone ===")
    best_hist_beta = tune_beta(hist_norm, "history")
    print(f"-> best beta={best_hist_beta['beta']}  MAP@5(TOP5)={best_hist_beta['top5']:.4f}  "
          f"(gain over beta=0: {best_hist_beta['top5']-hist_alone_top5:+.4f})")

    print("\n=== Stage 3: airing-frequency factor on PPR alone ===")
    best_ppr_beta = tune_beta(ppr_norm, "ppr")
    print(f"-> best beta={best_ppr_beta['beta']}  MAP@5(TOP5)={best_ppr_beta['top5']:.4f}  "
          f"(gain over beta=0: {best_ppr_beta['top5']-ppr_alone_top5:+.4f})")

    print(f"\n=== Stage 4: airing-frequency factor on HYBRID (w={best_w}) ===")
    best_hybrid_beta = tune_beta(hybrid_base, "hybrid")
    print(f"-> best beta={best_hybrid_beta['beta']}  MAP@5(TOP5)={best_hybrid_beta['top5']:.4f}  "
          f"(gain over beta=0: {best_hybrid_beta['top5']-best_w_top5:+.4f})")

    print("\n=== Component gain summary (TOP-5 preferred truth) ===")
    print(f"History alone                 : {hist_alone_top5:.4f}")
    print(f"PPR alone                      : {ppr_alone_top5:.4f}")
    print(f"Hybrid fusion (best w={best_w}) : {best_w_top5:.4f}  "
          f"(fusion gain over best single: {best_w_top5-max(hist_alone_top5,ppr_alone_top5):+.4f})")
    print(f"History + airing (best beta)   : {best_hist_beta['top5']:.4f}  "
          f"(airing gain: {best_hist_beta['top5']-hist_alone_top5:+.4f})")
    print(f"PPR + airing (best beta)       : {best_ppr_beta['top5']:.4f}  "
          f"(airing gain: {best_ppr_beta['top5']-ppr_alone_top5:+.4f})")
    print(f"Hybrid + airing (best beta)    : {best_hybrid_beta['top5']:.4f}  "
          f"(airing gain: {best_hybrid_beta['top5']-best_w_top5:+.4f})")

    overall = [
        ("History alone", hist_alone_all, hist_alone_top5),
        ("PPR alone", ppr_alone_all, ppr_alone_top5),
        ("Hybrid (w only)", best_w_all, best_w_top5),
        ("History+airing", best_hist_beta["all"], best_hist_beta["top5"]),
        ("PPR+airing", best_ppr_beta["all"], best_ppr_beta["top5"]),
        ("Hybrid+airing", best_hybrid_beta["all"], best_hybrid_beta["top5"]),
    ]
    best_overall = max(overall, key=lambda r: r[2])
    print(f"\n-> BEST OVERALL: {best_overall[0]}  MAP@5(TOP5)={best_overall[2]:.4f}")

    pd.DataFrame(results).to_csv(DATA_DIR / "results_hybrid_grid.csv", index=False)
    pd.DataFrame([{"method": "Hybrid[best]", "w": best_w, "beta": best_hybrid_beta["beta"],
                   "use_airing": best_hybrid_beta["top5"] > best_w_top5,
                   "map_all": max(best_w_all, best_hybrid_beta["all"]) if best_hybrid_beta["top5"] > best_w_top5 else best_w_all,
                   "map_top5": best_overall[2]}]).to_csv(DATA_DIR / "results_hybrid_best.csv", index=False)

    print(f"\ntotal runtime: {time.time()-t0_all:.1f}s")


if __name__ == "__main__":
    main()
