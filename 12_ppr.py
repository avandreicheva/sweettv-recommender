"""
Personalized PageRank on a show co-view graph built from TRAIN_A.
  - edge weight(i,j) = number of TRAIN_A users who watched both i and j
    (binary presence, not weighted by seconds)
  - transition matrix P = row-normalized co-view matrix (isolated items get
    a self-loop so the walk stays well-defined)
  - restart vector r_u = user's TRAIN_A watched_seconds, normalized to sum 1
    over items in the graph vocabulary
  - pi_u solves pi = alpha * P^T @ pi + (1-alpha) * r_u, i.e.
    (I - alpha*P^T) @ pi = (1-alpha) * r_u, solved for ALL users at once
    via one shared LU factorization (multiple right-hand sides)
alpha tuned on validation (TOP5 ground truth). Seed=42 (tie-break jitter).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR


def build_transition_matrix(train_a, item_index):
    n_users_u = train_a["user_id"].unique()
    user_idx_map = {u: i for i, u in enumerate(n_users_u)}
    rows = train_a["user_id"].map(user_idx_map).to_numpy()
    cols = train_a["tv_show_id"].map(item_index).to_numpy()
    n_items = len(item_index)
    M = sp.csr_matrix((np.ones(len(train_a)), (rows, cols)), shape=(len(n_users_u), n_items))
    M.data[:] = 1.0  # binary presence (collapse duplicate entries, if any)

    co_view = (M.T @ M).toarray()
    np.fill_diagonal(co_view, 0.0)
    row_sums = co_view.sum(axis=1)
    isolated = row_sums == 0
    row_sums[isolated] = 1.0
    P = co_view / row_sums[:, None]
    P[isolated, isolated] = 1.0  # self-loop for isolated nodes
    return P


def build_restart_matrix(train_a, item_index, users_in_vocab):
    n_items = len(item_index)
    user_pos = {u: i for i, u in enumerate(users_in_vocab)}
    sub = train_a[train_a["user_id"].isin(user_pos)]
    rows = sub["tv_show_id"].map(item_index).to_numpy()
    cols = sub["user_id"].map(user_pos).to_numpy()
    vals = sub["watched_seconds"].to_numpy()
    R = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, len(users_in_vocab))).toarray()
    col_sums = R.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    R = R / col_sums[None, :]
    return R


def main():
    t_all0 = time.time()
    data = ce.load_core()
    train_a, pool_set = data["train_a"], data["pool_set"]
    population, gt_all, gt_top5 = data["population"], data["gt_all"], data["gt_top5"]
    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")

    item_ids = sorted(train_a["tv_show_id"].unique())
    item_index = {s: i for i, s in enumerate(item_ids)}
    pool_items_in_vocab = [s for s in item_ids if s in pool_set]
    pool_rows = np.array([item_index[s] for s in pool_items_in_vocab])
    pool_show_ids = np.array(pool_items_in_vocab)

    users_in_vocab = sorted(train_a["user_id"].unique())
    user_pos = {u: i for i, u in enumerate(users_in_vocab)}

    print(f"graph: {len(item_ids)} item nodes, {len(users_in_vocab)} users with history, "
          f"pool items in vocab: {len(pool_items_in_vocab)}/{len(pool_set)}")

    t0 = time.time()
    P = build_transition_matrix(train_a, item_index)
    R = build_restart_matrix(train_a, item_index, users_in_vocab)
    print(f"graph build time: {time.time()-t0:.1f}s  (P={P.shape}, R={R.shape})")

    n = P.shape[0]
    I = np.eye(n)

    grid_alpha = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = []
    best = None

    for alpha in grid_alpha:
        t0 = time.time()
        A = I - alpha * P.T
        B = (1 - alpha) * R
        Pi = la.solve(A, B)  # n_items x n_users_in_vocab, one shared LU factorization
        solve_time = time.time() - t0

        preds = {}
        for u in population:
            upos = user_pos.get(u)
            if upos is None:
                preds[u] = popularity_ranked[: ce.K].tolist()
                continue
            scores = Pi[pool_rows, upos]
            top_local = np.argpartition(-scores, min(ce.K, len(scores) - 1))[: ce.K]
            top_local = top_local[np.argsort(-scores[top_local])]
            rec = pool_show_ids[top_local].tolist()
            preds[u] = ce.fill_with_popularity(rec, popularity_ranked, k=ce.K)

        res_all, res_top5 = ce.evaluate_both(preds, gt_all, gt_top5)
        dt = time.time() - t0
        print(f"  alpha={alpha:.2f}  MAP@5(ALL)={res_all['map@k']:.4f} "
              f"MAP@5(TOP5)={res_top5['map@k']:.4f}  solve={solve_time:.1f}s total={dt:.1f}s")
        results.append({"alpha": alpha, "map_all": res_all["map@k"],
                         "map_top5": res_top5["map@k"], "time_s": dt})
        if best is None or res_top5["map@k"] > best["map_top5"]:
            best = results[-1]

    print(f"\n-> best PPR config: alpha={best['alpha']} "
          f"MAP@5(ALL)={best['map_all']:.4f} MAP@5(TOP5)={best['map_top5']:.4f}")
    print(f"total runtime: {time.time()-t_all0:.1f}s")

    pd.DataFrame(results).to_csv(DATA_DIR / "results_ppr_grid.csv", index=False)
    pd.DataFrame([{"method": f"PPR[alpha={best['alpha']}]", "map_all": best["map_all"],
                   "map_top5": best["map_top5"], "time_s": best["time_s"]}]
                 ).to_csv(DATA_DIR / "results_ppr_best.csv", index=False)


if __name__ == "__main__":
    main()
