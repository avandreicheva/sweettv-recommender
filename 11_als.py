"""
Implicit-feedback ALS (Hu, Koren, Volinsky 2008), implemented from scratch
with numpy/scipy. Confidence c_ui = 1 + a*log(1+watched_seconds). Trained on
TRAIN_A. factors and a tuned on validation (TOP5 ground truth); reg and
iters held fixed. Seed=42.
"""
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
import common_eval as ce

DATA_DIR = ce.DATA_DIR
REG = 0.1
ITERS = 10


def build_conf_csr(train_a, user_index, item_index, alpha):
    rows = train_a["user_id"].map(user_index).to_numpy()
    cols = train_a["tv_show_id"].map(item_index).to_numpy()
    conf = 1.0 + alpha * np.log1p(train_a["watched_seconds"].to_numpy())
    mat = sp.csr_matrix((conf, (rows, cols)), shape=(len(user_index), len(item_index)))
    return mat


def als_fit(conf_csr, n_factors, reg=REG, iters=ITERS, seed=ce.SEED):
    n_users, n_items = conf_csr.shape
    rng = np.random.default_rng(seed)
    U = rng.normal(scale=0.01, size=(n_users, n_factors))
    V = rng.normal(scale=0.01, size=(n_items, n_factors))
    conf_item_major = conf_csr.T.tocsr()
    I = np.eye(n_factors)

    for _ in range(iters):
        YtY = V.T @ V + reg * I
        for u in range(n_users):
            s, e = conf_csr.indptr[u], conf_csr.indptr[u + 1]
            if s == e:
                continue
            idx = conf_csr.indices[s:e]
            c = conf_csr.data[s:e]
            Vu = V[idx]
            A = YtY + Vu.T @ (Vu * (c - 1.0)[:, None])
            b = Vu.T @ c
            U[u] = np.linalg.solve(A, b)

        XtX = U.T @ U + reg * I
        for i in range(n_items):
            s, e = conf_item_major.indptr[i], conf_item_major.indptr[i + 1]
            if s == e:
                continue
            idx = conf_item_major.indices[s:e]
            c = conf_item_major.data[s:e]
            Uu = U[idx]
            A = XtX + Uu.T @ (Uu * (c - 1.0)[:, None])
            b = Uu.T @ c
            V[i] = np.linalg.solve(A, b)

    return U, V


def main():
    t_all0 = time.time()
    data = ce.load_core()
    train_a, pool_set = data["train_a"], data["pool_set"]
    population, gt_all, gt_top5 = data["population"], data["gt_all"], data["gt_top5"]
    popularity_ranked = ce.popularity_ranking(train_a, pool_set, by="total_watched_seconds")

    user_ids = sorted(train_a["user_id"].unique())
    item_ids = sorted(train_a["tv_show_id"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {s: i for i, s in enumerate(item_ids)}
    pool_items_in_vocab = [s for s in item_ids if s in pool_set]
    pool_cols = np.array([item_index[s] for s in pool_items_in_vocab])
    pool_show_ids = np.array(pool_items_in_vocab)

    print(f"TRAIN_A matrix: {len(user_ids)} users x {len(item_ids)} items, "
          f"{len(train_a)} nonzeros, pool items in vocab: {len(pool_items_in_vocab)}/{len(pool_set)}")

    grid_factors = [10, 20, 50]
    grid_alpha = [0.05, 0.1, 0.25, 0.5, 1, 2, 5]

    best = None
    results = []
    for f in grid_factors:
        for a in grid_alpha:
            t0 = time.time()
            conf_csr = build_conf_csr(train_a, user_index, item_index, a)
            U, V = als_fit(conf_csr, n_factors=f)
            fit_time = time.time() - t0

            scores = U @ V[pool_cols].T  # n_users x n_pool_items

            preds = {}
            for u in population:
                uidx = user_index.get(u)
                if uidx is None:
                    preds[u] = popularity_ranked[: ce.K].tolist()
                    continue
                row = scores[uidx]
                top_local = np.argpartition(-row, min(ce.K, len(row) - 1))[: ce.K]
                top_local = top_local[np.argsort(-row[top_local])]
                rec = pool_show_ids[top_local].tolist()
                preds[u] = ce.fill_with_popularity(rec, popularity_ranked, k=ce.K)

            res_all, res_top5 = ce.evaluate_both(preds, gt_all, gt_top5)
            dt = time.time() - t0
            print(f"  factors={f:3d} a={a:5.2f}  MAP@5(ALL)={res_all['map@k']:.4f} "
                  f"MAP@5(TOP5)={res_top5['map@k']:.4f}  fit={fit_time:.1f}s total={dt:.1f}s")
            results.append({"factors": f, "a": a, "map_all": res_all["map@k"],
                             "map_top5": res_top5["map@k"], "time_s": dt})
            if best is None or res_top5["map@k"] > best["map_top5"]:
                best = results[-1]

    print(f"\n-> best ALS config: factors={best['factors']} a={best['a']} "
          f"MAP@5(ALL)={best['map_all']:.4f} MAP@5(TOP5)={best['map_top5']:.4f}")
    print(f"total runtime: {time.time()-t_all0:.1f}s")

    pd.DataFrame(results).to_csv(DATA_DIR / "results_als_grid.csv", index=False)
    pd.DataFrame([{"method": f"ALS[factors={best['factors']},a={best['a']}]",
                   "map_all": best["map_all"], "map_top5": best["map_top5"],
                   "time_s": best["time_s"]}]).to_csv(DATA_DIR / "results_als_best.csv", index=False)


if __name__ == "__main__":
    main()
