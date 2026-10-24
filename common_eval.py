"""Shared data loading, ground-truth construction, and small utilities reused
by all method scripts (popularity, user-history, ALS, PPR)."""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
K = 5
SEED = 42

_spec = importlib.util.spec_from_file_location("evaluator", DATA_DIR / "scripts" / "08_evaluator.py")
evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluator)
map_at_k = evaluator.map_at_k


def build_ground_truths(val, pool_set, population):
    """ALL / TOP5 / TOP10 / TOP20 preferred-shows ground truths from VAL,
    pool-restricted, for an arbitrary population of users."""
    val_pool = val[val["tv_show_id"].isin(pool_set)]

    gt_all_map = val_pool.groupby("user_id")["tv_show_id"].apply(set).to_dict()
    gt_all = {u: gt_all_map.get(u, set()) for u in population}

    vs = val_pool.sort_values(
        ["user_id", "watched_seconds", "count", "tv_show_id"],
        ascending=[True, False, False, True],
    )

    def topn_gt(n):
        m = vs.groupby("user_id").head(n).groupby("user_id")["tv_show_id"].apply(set).to_dict()
        return {u: m.get(u, set()) for u in population}

    return {
        "val_pool": val_pool,
        "gt_all": gt_all,
        "gt_top5": topn_gt(K),
        "gt_top10": topn_gt(10),
        "gt_top20": topn_gt(20),
    }


def load_core():
    train_a = pd.read_parquet(DATA_DIR / "interactions_TRAIN_A.parquet")
    val = pd.read_parquet(DATA_DIR / "interactions_VAL.parquet")
    pool_set = set(pd.read_csv(DATA_DIR / "candidate_pool_VAL.csv")["tv_show_id"].tolist())
    population = set(train_a["user_id"].unique()) | set(val["user_id"].unique())

    gts = build_ground_truths(val, pool_set, population)
    return {
        "train_a": train_a,
        "val": val,
        "val_pool": gts["val_pool"],
        "pool_set": pool_set,
        "population": population,
        "gt_all": gts["gt_all"],
        "gt_top5": gts["gt_top5"],
        "gt_top10": gts["gt_top10"],
        "gt_top20": gts["gt_top20"],
    }


def popularity_ranking(train_a, pool_set, by="distinct_viewers", seed=SEED):
    """Returns full ranked array of pool show_ids, most-preferred first."""
    sub = train_a[train_a["tv_show_id"].isin(pool_set)]
    if by == "distinct_viewers":
        scores = sub.groupby("tv_show_id")["user_id"].nunique()
    elif by == "total_watched_seconds":
        scores = sub.groupby("tv_show_id")["watched_seconds"].sum()
    else:
        raise ValueError(by)
    scores = scores.reindex(sorted(pool_set), fill_value=0)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(scores)) * 1e-6
    order = (scores.to_numpy() + jitter).argsort()[::-1]
    return scores.index.to_numpy()[order]


def fill_with_popularity(rec, popularity_ranked, k=K):
    seen = set(rec)
    out = list(rec)
    for show in popularity_ranked:
        if len(out) == k:
            break
        if show not in seen:
            out.append(show)
            seen.add(show)
    return out


def evaluate_both(preds: dict, gt_all: dict, gt_top5: dict):
    res_all = map_at_k(preds, gt_all, k=K)
    res_top5 = map_at_k(preds, gt_top5, k=K)
    return res_all, res_top5


def evaluate_selection(preds: dict, gt_all: dict, gt_top10: dict, gt_top20: dict):
    """Selection criterion = mean MAP@5 over ALL, TOP-10 and TOP-20 preferred
    truths (replaces the TOP-5-only criterion, which Kaggle showed mis-ranks
    methods)."""
    r_all = map_at_k(preds, gt_all, k=K)
    r_10 = map_at_k(preds, gt_top10, k=K)
    r_20 = map_at_k(preds, gt_top20, k=K)
    mean_score = (r_all["map@k"] + r_10["map@k"] + r_20["map@k"]) / 3.0
    return {"map_all": r_all["map@k"], "map_top10": r_10["map@k"],
            "map_top20": r_20["map@k"], "mean_score": mean_score}


def fast_rank_to_preds(score_mat, user_order, pool_items, population, popularity_ranked, k=K):
    """Vectorised equivalent of rank_scores_to_preds: sorts every row once
    with numpy instead of looping per user, then only loops (cheaply) to
    assemble the popularity-filled top-k lists. Used for large grid searches."""
    pool_arr = np.array(pool_items)
    user_index = {u: i for i, u in enumerate(user_order)}
    order = np.argsort(-score_mat, axis=1)
    nz_counts = (score_mat > 1e-15).sum(axis=1)
    pop_list = popularity_ranked.tolist() if hasattr(popularity_ranked, "tolist") else list(popularity_ranked)

    preds = {}
    for u in population:
        uidx = user_index.get(u)
        if uidx is None or nz_counts[uidx] == 0:
            preds[u] = pop_list[:k]
            continue
        take = int(min(k, nz_counts[uidx]))
        rec = pool_arr[order[uidx, :take]].tolist()
        preds[u] = rec if take == k else fill_with_popularity(rec, popularity_ranked, k=k)
    return preds


# ---------------------------------------------------------------------------
# Reusable scoring building blocks for the hybrid (history + PPR) model and
# for the final TRAIN_FULL refit. All take an arbitrary interactions
# dataframe + pool so they work for both TRAIN_A->VAL tuning and the
# TRAIN_FULL->TEST final fit.
# ---------------------------------------------------------------------------

def airing_counts(schedule_path, pool_set, chunksize=500_000):
    """Number of labelled airings per tv_show_id in a schedule file, pool-restricted."""
    counts = {}
    for chunk in pd.read_csv(schedule_path, chunksize=chunksize, usecols=["tv_show_id"]):
        c = chunk[chunk["tv_show_id"].isin(pool_set)]
        vc = c["tv_show_id"].value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    return counts


def airing_factor_array(pool_items, airing_count_map, beta):
    counts = np.array([airing_count_map.get(s, 1) for s in pool_items], dtype="float64")
    return np.power(1.0 + np.log(counts), beta)


def history_score_matrix(df, pool_set, user_order, basis="both", half_life=14):
    """Dense (n_users x n_pool_items) recency-weighted history score matrix,
    row order = user_order, col order = sorted(pool_set). Rows for users not
    present in df are all-zero."""
    pool_items = sorted(pool_set)
    item_index = {s: i for i, s in enumerate(pool_items)}
    user_index = {u: i for i, u in enumerate(user_order)}

    hist = df[df["tv_show_id"].isin(pool_set) & df["user_id"].isin(user_index)].copy()
    anchor_t = df["last_watch_time"].max()
    if half_life is None:
        decay = 1.0
    else:
        days_since = (anchor_t - hist["last_watch_time"]) / 86400.0
        decay = np.power(0.5, days_since / half_life)

    if basis == "seconds":
        raw = hist["watched_seconds"].to_numpy()
    elif basis == "count":
        raw = hist["count"].to_numpy()
    elif basis == "both":
        raw = hist["watched_seconds"].to_numpy() * hist["count"].to_numpy()
    else:
        raise ValueError(basis)

    rows = hist["user_id"].map(user_index).to_numpy()
    cols = hist["tv_show_id"].map(item_index).to_numpy()
    vals = raw * decay

    import scipy.sparse as sp
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_order), len(pool_items))).toarray()
    return mat, pool_items


def ppr_score_matrix(df, pool_set, user_order, alpha, restart_col="watched_seconds"):
    """Dense (n_users x n_pool_items) PPR score matrix. Graph vocabulary =
    all distinct tv_show_id in df. user_order rows not present in df -> zero.
    Items in pool_set but absent from df's vocabulary -> zero column.
    restart_col selects which interaction column weights the restart vector
    (e.g. "watched_seconds" or "count")."""
    import scipy.sparse as sp
    import scipy.linalg as la

    item_ids = sorted(df["tv_show_id"].unique())
    item_index = {s: i for i, s in enumerate(item_ids)}
    n_items = len(item_ids)

    df_users = sorted(df["user_id"].unique())
    df_user_pos = {u: i for i, u in enumerate(df_users)}

    rows = df["user_id"].map(df_user_pos).to_numpy()
    cols = df["tv_show_id"].map(item_index).to_numpy()
    M = sp.csr_matrix((np.ones(len(df)), (rows, cols)), shape=(len(df_users), n_items))
    M.data[:] = 1.0
    co_view = (M.T @ M).toarray()
    np.fill_diagonal(co_view, 0.0)
    row_sums = co_view.sum(axis=1)
    isolated = row_sums == 0
    row_sums[isolated] = 1.0
    P = co_view / row_sums[:, None]
    P[isolated, isolated] = 1.0

    R = sp.csr_matrix((df[restart_col].to_numpy().astype("float64"), (cols, rows)),
                       shape=(n_items, len(df_users))).toarray()
    col_sums = R.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    R = R / col_sums[None, :]

    I = np.eye(n_items)
    Pi = la.solve(I - alpha * P.T, (1 - alpha) * R)  # n_items x n_df_users

    pool_items = sorted(pool_set)
    pool_rows = np.array([item_index.get(s, -1) for s in pool_items])
    in_vocab = pool_rows >= 0

    out = np.zeros((len(user_order), len(pool_items)))
    user_index = {u: i for i, u in enumerate(user_order)}
    common_users = [u for u in df_users if u in user_index]
    if common_users:
        src_cols = [df_user_pos[u] for u in common_users]
        dst_rows = [user_index[u] for u in common_users]
        sub = Pi[np.ix_(pool_rows[in_vocab], src_cols)].T  # n_common_users x n_in_vocab
        out[np.ix_(dst_rows, np.flatnonzero(in_vocab))] = sub
    return out, pool_items


def normalize_rows(mat):
    sums = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return mat / sums


def rank_scores_to_preds(score_mat, user_order, pool_items, population, popularity_ranked, k=K):
    user_index = {u: i for i, u in enumerate(user_order)}
    pool_arr = np.array(pool_items)
    preds = {}
    for u in population:
        uidx = user_index.get(u)
        if uidx is None:
            preds[u] = popularity_ranked[:k].tolist()
            continue
        row = score_mat[uidx]
        nz = np.flatnonzero(row > 1e-15)
        if len(nz) == 0:
            preds[u] = popularity_ranked[:k].tolist()
            continue
        order = nz[np.argsort(-row[nz])][:k]
        rec = pool_arr[order].tolist()
        preds[u] = fill_with_popularity(rec, popularity_ranked, k=k)
    return preds


def rank_with_chain(mats, user_order, pool_items, population, popularity_ranked, k=K):
    """Tiered fallback ranking: fill from mats[0]'s nonzero entries first,
    then mats[1], etc (skipping already-chosen items), and finally from
    popularity_ranked if still short of k."""
    user_index = {u: i for i, u in enumerate(user_order)}
    pool_arr = np.array(pool_items)
    preds = {}
    for u in population:
        uidx = user_index.get(u)
        rec, seen = [], set()
        if uidx is not None:
            for mat in mats:
                if len(rec) == k:
                    break
                row = mat[uidx]
                nz = np.flatnonzero(row > 1e-15)
                nz = nz[~np.isin(pool_arr[nz], list(seen))] if seen else nz
                if len(nz) == 0:
                    continue
                order = nz[np.argsort(-row[nz])]
                for idx in order:
                    show = pool_arr[idx]
                    if show not in seen:
                        rec.append(show)
                        seen.add(show)
                        if len(rec) == k:
                            break
        preds[u] = fill_with_popularity(rec, popularity_ranked, k=k)
    return preds


def scheduled_seconds_map(schedule_paths, show_ids=None, chunksize=500_000):
    """Total scheduled duration (seconds) per labelled tv_show_id across one
    or more schedule files."""
    if isinstance(schedule_paths, (str, Path)):
        schedule_paths = [schedule_paths]
    totals = {}
    for p in schedule_paths:
        for chunk in pd.read_csv(p, chunksize=chunksize, usecols=["tv_show_id", "duration"]):
            c = chunk[chunk["tv_show_id"] != 0]
            if show_ids is not None:
                c = c[c["tv_show_id"].isin(show_ids)]
            if len(c) == 0:
                continue
            g = c.groupby("tv_show_id")["duration"].sum()
            for show, v in g.items():
                totals[show] = totals.get(show, 0.0) + float(v)
    return totals


def expected_watch_time_matrix(df, pool_set, user_order, train_schedule_paths,
                                target_schedule_paths, engagement_mode="ratio",
                                half_life=14, beta=1.0):
    """score(u,i) = engagement(u,i) * airtime(i, target window)^beta
    engagement = watched_seconds(u,i) [/ show's total scheduled airtime in the
    user's training window, if mode='ratio'], recency-decayed."""
    pool_items = sorted(pool_set)
    item_index = {s: i for i, s in enumerate(pool_items)}
    user_index = {u: i for i, u in enumerate(user_order)}

    sub = df[df["tv_show_id"].isin(pool_set) & df["user_id"].isin(user_index)].copy()
    anchor_t = df["last_watch_time"].max()
    if half_life is None:
        decay = 1.0
    else:
        days_since = (anchor_t - sub["last_watch_time"]) / 86400.0
        decay = np.power(0.5, days_since / half_life)

    if engagement_mode == "seconds":
        raw = sub["watched_seconds"].to_numpy()
    elif engagement_mode == "ratio":
        train_airtime = scheduled_seconds_map(train_schedule_paths,
                                               show_ids=set(sub["tv_show_id"].unique()))
        denom = sub["tv_show_id"].map(train_airtime).fillna(0.0).to_numpy()
        denom = np.where(denom > 0, denom, 1.0)
        raw = sub["watched_seconds"].to_numpy() / denom
    else:
        raise ValueError(engagement_mode)

    rows = sub["user_id"].map(user_index).to_numpy()
    cols = sub["tv_show_id"].map(item_index).to_numpy()
    vals = raw * decay

    import scipy.sparse as sp
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_order), len(pool_items))).toarray()

    target_airtime = scheduled_seconds_map(target_schedule_paths, show_ids=pool_set)
    airtime_arr = np.array([target_airtime.get(s, 0.0) for s in pool_items])
    factor = np.ones_like(airtime_arr) if beta == 0 else np.power(np.maximum(airtime_arr, 1.0), beta)
    mat = mat * factor[None, :]
    return mat, pool_items
