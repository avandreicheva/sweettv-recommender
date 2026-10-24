"""
Case studies for the report: one heavy and one light viewer among the
submission users. For each, compute Top-5 from popularity, user-history,
ALS, PPR, and the final hybrid (v5 settings) in the VALIDATION setup
(fit on TRAIN_A = weeks 11-20, scored against VAL = weeks 21-30), next to
what they actually watched in weeks 21-30 (top by count), plus their real
final Top-5 from submission_hybrid_v5.csv. Also prints the hybrid's
validation AP@5 for the heavy user with enough detail to check by hand.

User selection:
  HEAVY = 13571199046623737884: most distinct shows in TRAIN_A among
          submission users (445 shows, 7.24M watched seconds) -> strongest
          possible personalization signal.
  LIGHT = 8150816946177670084: only 1 distinct show in TRAIN_A (2753s) but
          still has VAL-period activity (5 shows) to score against ->
          sparsest realistic training signal, showing fallback behaviour.
"""
import importlib.util
import numpy as np
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR
HEAVY_USER = 13571199046623737884
LIGHT_USER = 8150816946177670084

_spec = importlib.util.spec_from_file_location("als", DATA_DIR / "scripts" / "11_als.py")
als_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(als_mod)

HIST_BASIS, HIST_HL = "both", 14  # the standalone "user-history" baseline
PPR_ALPHA = 0.8
ALS_FACTORS, ALS_A = 20, 0.05
V5 = dict(alpha=0.8, beta=2, w=0.6, half_life=28, basis="count", restart_col="count")


def load_title_map():
    m = pd.read_csv(DATA_DIR / "show_id_title_category.csv")
    return {row.tv_show_id: (row.title, row.category) for row in m.itertuples()}


def annotate(ids, title_map):
    out = []
    for i in ids:
        t, c = title_map.get(i, ("<unknown>", "<unknown>"))
        out.append(f"{i} | {t[:50]} | {c}")
    return out


def main():
    title_map = load_title_map()

    train_a = pd.read_parquet(DATA_DIR / "interactions_TRAIN_A.parquet")
    val = pd.read_parquet(DATA_DIR / "interactions_VAL.parquet")
    pool_set = set(pd.read_csv(DATA_DIR / "candidate_pool_VAL.csv")["tv_show_id"].tolist())
    population = set(train_a["user_id"].unique()) | set(val["user_id"].unique())
    gts = ce.build_ground_truths(val, pool_set, population)
    user_order = sorted(population)

    print(f"HEAVY user {HEAVY_USER}: "
          f"{train_a[train_a.user_id==HEAVY_USER]['tv_show_id'].nunique()} shows in TRAIN_A, "
          f"{train_a[train_a.user_id==HEAVY_USER]['watched_seconds'].sum():.0f}s watched, "
          f"{val[val.user_id==HEAVY_USER]['tv_show_id'].nunique()} shows in VAL")
    print(f"LIGHT user {LIGHT_USER}: "
          f"{train_a[train_a.user_id==LIGHT_USER]['tv_show_id'].nunique()} shows in TRAIN_A, "
          f"{train_a[train_a.user_id==LIGHT_USER]['watched_seconds'].sum():.0f}s watched, "
          f"{val[val.user_id==LIGHT_USER]['tv_show_id'].nunique()} shows in VAL\n")

    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")

    # ---- popularity ----
    pop_top5 = popularity_ranked[: ce.K].tolist()

    # ---- user-history ----
    hist_mat, pool_items = ce.history_score_matrix(train_a, pool_set, user_order, basis=HIST_BASIS, half_life=HIST_HL)
    preds_hist = ce.rank_scores_to_preds(hist_mat, user_order, pool_items, [HEAVY_USER, LIGHT_USER], popularity_ranked)

    # ---- ALS ----
    user_ids = sorted(train_a["user_id"].unique())
    item_ids = sorted(train_a["tv_show_id"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {s: i for i, s in enumerate(item_ids)}
    pool_items_in_vocab = [s for s in item_ids if s in pool_set]
    pool_cols = np.array([item_index[s] for s in pool_items_in_vocab])
    pool_show_ids = np.array(pool_items_in_vocab)
    conf_csr = als_mod.build_conf_csr(train_a, user_index, item_index, ALS_A)
    U, V = als_mod.als_fit(conf_csr, n_factors=ALS_FACTORS)
    scores = U @ V[pool_cols].T
    preds_als = {}
    for u in [HEAVY_USER, LIGHT_USER]:
        uidx = user_index.get(u)
        if uidx is None:
            preds_als[u] = popularity_ranked[: ce.K].tolist()
            continue
        row = scores[uidx]
        top_local = np.argpartition(-row, min(ce.K, len(row) - 1))[: ce.K]
        top_local = top_local[np.argsort(-row[top_local])]
        rec = pool_show_ids[top_local].tolist()
        preds_als[u] = ce.fill_with_popularity(rec, popularity_ranked, k=ce.K)

    # ---- PPR ----
    ppr_mat, pool_items_p = ce.ppr_score_matrix(train_a, pool_set, user_order, alpha=PPR_ALPHA)
    preds_ppr = ce.rank_scores_to_preds(ppr_mat, user_order, pool_items_p, [HEAVY_USER, LIGHT_USER], popularity_ranked)

    # ---- Hybrid v5 (validation setup) ----
    ppr_mat5, pool_items5 = ce.ppr_score_matrix(train_a, pool_set, user_order, alpha=V5["alpha"],
                                                 restart_col=V5["restart_col"])
    ppr_norm5 = ce.normalize_rows(ppr_mat5)
    hist_mat5, _ = ce.history_score_matrix(train_a, pool_set, user_order, basis=V5["basis"], half_life=V5["half_life"])
    hist_norm5 = ce.normalize_rows(hist_mat5)
    airing_map = ce.airing_counts(DATA_DIR / "export_arh_21-30-final.csv", pool_set)
    factor = ce.airing_factor_array(pool_items5, airing_map, V5["beta"])
    combined5 = V5["w"] * (hist_norm5 * factor[None, :]) + (1 - V5["w"]) * (ppr_norm5 * factor[None, :])
    preds_hybrid = ce.rank_scores_to_preds(combined5, user_order, pool_items5, [HEAVY_USER, LIGHT_USER], popularity_ranked)

    # ---- actual VAL shows, top by count ----
    def val_top_by_count(u, n=10):
        sub = val[(val.user_id == u) & (val.tv_show_id.isin(pool_set))]
        sub = sub.sort_values(["count", "watched_seconds"], ascending=[False, False])
        return sub["tv_show_id"].head(n).tolist()

    # ---- final submission_hybrid_v5.csv row ----
    sub_v5 = pd.read_csv(DATA_DIR / "submission_hybrid_v5.csv")
    sub_v5_map = {row.user_id: [int(x) for x in row.tv_show_id.split()] for row in sub_v5.itertuples()}

    for u, label in [(HEAVY_USER, "HEAVY"), (LIGHT_USER, "LIGHT")]:
        print(f"\n{'='*100}\n{label} VIEWER: user_id={u}\n{'='*100}")
        print(f"\n-- Popularity Top-5 --")
        for row in annotate(pop_top5, title_map):
            print(f"   {row}")
        print(f"\n-- User-history Top-5 (basis=both, hl=14) --")
        for row in annotate(preds_hist[u], title_map):
            print(f"   {row}")
        print(f"\n-- ALS Top-5 (factors=20, a=0.05) --")
        for row in annotate(preds_als[u], title_map):
            print(f"   {row}")
        print(f"\n-- PPR Top-5 (alpha=0.8) --")
        for row in annotate(preds_ppr[u], title_map):
            print(f"   {row}")
        print(f"\n-- Final Hybrid (v5 settings) Top-5, VALIDATION setup (fit wk11-20, score wk21-30) --")
        for row in annotate(preds_hybrid[u], title_map):
            print(f"   {row}")
        print(f"\n-- Actually watched in weeks 21-30, top by count --")
        for row in annotate(val_top_by_count(u), title_map):
            print(f"   {row}")
        print(f"\n-- Final Top-5 from submission_hybrid_v5.csv (TEST period, weeks 31-42) --")
        for row in annotate(sub_v5_map[u], title_map):
            print(f"   {row}")

        if label == "HEAVY":
            r_u = gts["gt_all"][u]
            print(f"\n-- AP@5 by hand for HEAVY user's hybrid validation Top-5 --")
            print(f"R_u (all pool-restricted VAL shows watched) size = {len(r_u)}  ->  min(5,|R_u|) = {min(5,len(r_u))}")
            print(f"Hybrid Top-5: {preds_hybrid[u]}")
            hits = 0
            terms = []
            for rank, item in enumerate(preds_hybrid[u], start=1):
                is_hit = item in r_u
                if is_hit:
                    hits += 1
                precision_at_rank = hits / rank if is_hit else 0.0
                terms.append(precision_at_rank)
                print(f"   rank {rank}: {item}  hit={is_hit}  "
                      f"{'precision@'+str(rank)+'='+str(hits)+'/'+str(rank)+'='+f'{precision_at_rank:.4f}' if is_hit else '(miss, contributes 0)'}")
            manual_ap = sum(terms) / min(5, len(r_u))
            evaluator_ap = ce.evaluator.average_precision_at_k(preds_hybrid[u], r_u, k=5)
            print(f"   manual AP@5 = sum({[f'{t:.4f}' for t in terms if t>0]}) / {min(5,len(r_u))} = {manual_ap:.4f}")
            print(f"   evaluator.average_precision_at_k(...) = {evaluator_ap:.4f}")
            assert abs(manual_ap - evaluator_ap) < 1e-9, "manual vs evaluator AP mismatch!"
            print("   MATCH confirmed.")


if __name__ == "__main__":
    main()
