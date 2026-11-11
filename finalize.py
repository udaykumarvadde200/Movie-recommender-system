"""
Final step: popularity re-rank + compact top-20 neighbour file for the app.

    python finalize.py --csv data/tmdb_5000_movies.csv
    python finalize.py --csv data/tmdb_5000_movies.csv --beta 0.3     # more popularity weight

For each movie: take the 50 best candidates by hybrid similarity, then re-order them by
    (1 - beta) * similarity(0-1 within the pool) + beta * popularity(0-1)
where popularity is the percentile of the IMDB weighted rating (uses vote_average + vote_count).
Writes neighbors.npz (~0.5 MB) which app.py loads instead of the 85 MB matrix.
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="tmdb_5000_movies.csv")
ap.add_argument("--sim", default="models/sim_hybrid.npy")
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--pool", type=int, default=50)
ap.add_argument("--keep", type=int, default=20)
args = ap.parse_args()

K, N, SEED = 10, 500, 42
movies = pd.read_pickle("movies.pkl")
genres = [set(g) for g in movies["genres"]]

# ---- popularity: IMDB weighted rating ----
raw = pd.read_csv(args.csv)[["id", "vote_average", "vote_count"]].drop_duplicates("id")
df = movies[["movie_id"]].merge(raw, left_on="movie_id", right_on="id", how="left")
assert len(df) == len(movies), "row mismatch while merging vote data"
v = df["vote_count"].fillna(0).to_numpy(float)
R = df["vote_average"].fillna(0).to_numpy(float)
C = R[v > 0].mean()
m = np.quantile(v, 0.6)
wr = v / (v + m) * R + m / (v + m) * C
pop = pd.Series(wr).rank(pct=True).to_numpy()

# ---- re-rank ----
sim = np.load(args.sim).astype("float32")
np.fill_diagonal(sim, -np.inf)
n = len(movies)
idx = np.zeros((n, args.keep), dtype=np.int16)
score = np.zeros((n, args.keep), dtype=np.float32)
base_top = np.zeros((n, K), dtype=np.int64)
for i in range(n):
    pool = np.argpartition(sim[i], -args.pool)[-args.pool:]
    s = sim[i][pool]
    s01 = (s - s.min()) / (s.max() - s.min() + 1e-9)
    final = (1 - args.beta) * s01 + args.beta * pop[pool]
    order = np.argsort(-final)[: args.keep]
    idx[i], score[i] = pool[order], s01[order]
    base_top[i] = pool[np.argsort(-s)[:K]]  # plain hybrid top-K, for comparison

np.savez_compressed("neighbors.npz", idx=idx, score=score)
print(f"neighbors.npz written: {os.path.getsize('neighbors.npz') / 1e6:.2f} MB")

# ---- compare plain hybrid vs re-ranked ----
rng = np.random.default_rng(SEED)
queries = rng.choice(n, size=min(N, n), replace=False)


def metrics(tops):
    prec, jac, seen, avgpop = [], [], set(), []
    for q in queries:
        top = tops[q][:K]
        seen.update(int(t) for t in top)
        avgpop.append(pop[top].mean())
        base = genres[q]
        if not base:
            continue
        prec.append(np.mean([len(base & genres[j]) > 0 for j in top]))
        jac.append(np.mean([len(base & genres[j]) / len(base | genres[j]) for j in top]))
    return {
        f"Precision@{K}": round(np.mean(prec), 3),
        f"Jaccard@{K}": round(np.mean(jac), 3),
        "Coverage": round(len(seen) / n, 3),
        "AvgPopularity": round(np.mean(avgpop), 3),
    }


table = pd.DataFrame(
    [{"model": "hybrid", **metrics(base_top)}, {"model": f"hybrid+rerank(b={args.beta})", **metrics(idx.astype(int))}]
)
print(table.to_string(index=False))

for t in ["Avatar", "The Dark Knight", "Toy Story", "Titanic"]:
    if t in set(movies["title"]):
        i = movies.index[movies["title"] == t][0]
        print(f"{t}: ", ", ".join(movies["title"].iloc[idx[i][:6]]))
