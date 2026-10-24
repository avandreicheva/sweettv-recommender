"""
Streaming stats for the view log (dataset11-30.csv):
- distinct users, distinct channels (vsetv_id)
- duration quantiles, computed via a streaming histogram (exact quantiles
  would require sorting 5.8M values; we use a bounded-resolution histogram
  which is sufficient for threshold-setting purposes) plus exact min/max/mean.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
PATH = DATA_DIR / "dataset11-30.csv"
CHUNKSIZE = 500_000

# log-scale histogram bins from 1s to ~24h, fine enough for quantile estimation
BIN_EDGES = np.unique(np.concatenate([[0], np.logspace(0, np.log10(24 * 3600), 400)]))


def main():
    users = set()
    channels = set()
    n = 0
    total = 0.0
    sqtotal = 0.0
    dmin = None
    dmax = None
    hist = np.zeros(len(BIN_EDGES) - 1, dtype=np.int64)
    neg_or_zero = 0

    for chunk in pd.read_csv(
        PATH,
        chunksize=CHUNKSIZE,
        usecols=["user_id", "vsetv_id", "duraton"],
        dtype={"user_id": "int64", "vsetv_id": "int32", "duraton": "int64"},
    ):
        users.update(chunk["user_id"].unique().tolist())
        channels.update(chunk["vsetv_id"].unique().tolist())
        d = chunk["duraton"].to_numpy()
        n += len(d)
        total += d.sum()
        sqtotal += np.square(d, dtype=np.float64).sum()
        cmin, cmax = d.min(), d.max()
        dmin = cmin if dmin is None or cmin < dmin else dmin
        dmax = cmax if dmax is None or cmax > dmax else dmax
        neg_or_zero += int((d <= 0).sum())
        hist += np.histogram(np.clip(d, 0, None), bins=BIN_EDGES)[0]

    mean = total / n
    var = sqtotal / n - mean ** 2
    std = var ** 0.5

    # quantiles from histogram (approximate)
    cum = np.cumsum(hist)
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    approx_q = {}
    for q in qs:
        target = q * n
        idx = np.searchsorted(cum, target)
        idx = min(idx, len(BIN_EDGES) - 2)
        approx_q[q] = BIN_EDGES[idx + 1]

    print(f"rows={n}")
    print(f"distinct users={len(users)}")
    print(f"distinct channels (vsetv_id)={len(channels)}")
    print(f"duraton: min={dmin} max={dmax} mean={mean:.1f} std={std:.1f}")
    print(f"duraton<=0 count={neg_or_zero} ({100*neg_or_zero/n:.3f}%)")
    print("approx duraton quantiles (seconds):")
    for q, v in approx_q.items():
        print(f"  p{int(q*100):>2d} = {v:.0f}s  ({v/60:.1f} min)")


if __name__ == "__main__":
    main()
