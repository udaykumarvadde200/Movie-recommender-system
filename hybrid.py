"""
Blend the count (tag/genre/cast overlap) and embed (meaning of the plot) similarity matrices.

    python hybrid.py                 # sweep alpha, print metrics table
    python hybrid.py --alpha 0.5     # also save models/sim_hybrid.npy and similarity.pkl.gz (for the app)

score = alpha * z(count) + (1 - alpha) * z(embed)     (z = per-row standardisation)
alpha=1 -> pure count, alpha=0 -> pure embeddings.
"""
import argparse
import gzip
import pickle

import numpy as np
import pandas as pd

K, N, SEED = 10, 500, 42
ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=None, help="save a hybrid built with this weight")
args = ap.parse_args()

movies = pd.read_pickle("movies.pkl")
genres = [set(g) for g in movies["genres"]]
rng = np.random.default_rng(SEED)
queries = rng.choice(len(movies), size=min(N, len(movies)), replace=False)

count = np.load("models/sim_count.npy").astype("float32")
embed = np.load("models/sim_embed.npy").astype("float32")


def z(m):  # put both matrices on the same scale, row by row
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


zc, ze = z(count), z(embed)


def evaluate(sim):
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    prec, jac, seen = [], [], set()
    for q in queries:
        top = np.argpartition(sim[q], -K)[-K:]
        seen.update(top.tolist())
        base = genres[q]
        if not base:
            continue
        prec.append(np.mean([len(base & genres[i]) > 0 for i in top]))
        jac.append(np.mean([len(base & genres[i]) / len(base | genres[i]) for i in top]))
    return round(np.mean(prec), 3), round(np.mean(jac), 3), round(len(seen) / len(movies), 3)


rows = []
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    p, j, c = evaluate(a * zc + (1 - a) * ze)
    rows.append({"alpha": a, f"Precision@{K}": p, f"Jaccard@{K}": j, "Coverage": c})
print(pd.DataFrame(rows).to_string(index=False))

if args.alpha is not None:
    h = (args.alpha * zc + (1 - args.alpha) * ze).astype("float32")
    np.save("models/sim_hybrid.npy", h)
    with gzip.open("similarity.pkl.gz", "wb") as f:
        pickle.dump(h, f)
    print(f"\nSaved models/sim_hybrid.npy and similarity.pkl.gz (alpha={args.alpha}). Run compare.py to inspect.")
