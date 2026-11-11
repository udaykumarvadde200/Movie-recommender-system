"""
Compare recommenders using genre-based metrics.

Usage:  python evaluate.py            (uses movies.pkl + models/sim_*.npy)

Metrics (averaged over N sampled query movies, top-K recommendations):
  Precision@K : share of recommendations sharing >=1 genre with the query
  Jaccard@K   : mean genre-set Jaccard overlap (stricter)
  Coverage    : share of the catalogue that ever appears in a top-K list
This is a proxy (genre is not ground truth taste), so say so in your README.
"""
import glob
import os

import numpy as np
import pandas as pd

K, N, SEED = 10, 500, 42

movies = pd.read_pickle("movies.pkl")
genres = [set(g) for g in movies["genres"]]
rng = np.random.default_rng(SEED)
queries = rng.choice(len(movies), size=min(N, len(movies)), replace=False)


def evaluate(sim):
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)  # never recommend the movie itself
    prec, jac, seen = [], [], set()
    for q in queries:
        top = np.argpartition(sim[q], -K)[-K:]
        seen.update(top.tolist())
        base = genres[q]
        if not base:
            continue
        prec.append(np.mean([len(base & genres[i]) > 0 for i in top]))
        jac.append(np.mean([len(base & genres[i]) / len(base | genres[i]) for i in top]))
    return np.mean(prec), np.mean(jac), len(seen) / len(movies)


rows = []
for path in sorted(glob.glob("models/sim_*.npy")):
    name = os.path.basename(path)[4:-4]
    p, j, c = evaluate(np.load(path))
    rows.append({"model": name, f"Precision@{K}": round(p, 3), f"Jaccard@{K}": round(j, 3), "Coverage": round(c, 3)})

print(pd.DataFrame(rows).to_string(index=False))
print("\nPaste this table into your README.")
