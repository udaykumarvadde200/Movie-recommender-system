"""
Step 2: item-item collaborative filtering vs content hybrid vs popularity, on held-out ratings.

    python cf_evaluate.py                 # needs data/cf_train.csv, data/cf_test.csv (from cf_prepare.py)
                                          # and models/sim_hybrid.npy (from hybrid.py --alpha 0.5)

Protocol
- "Liked" = rating >= 4 (train signal and test ground truth).
- For each test user, score every catalogue movie they have not rated in train, take the top 10.
- Metrics over users with at least one liked test movie: Precision@10, Recall@10, NDCG@10, Coverage.

Models
- popularity : number of train ratings per movie
- cf         : item-item cosine similarity on train ratings (with shrinkage for low overlap)
- content    : your hybrid content similarity (sim_hybrid.npy)
- blend a    : a * z(cf score) + (1 - a) * z(content score), per user
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--neighbors", type=int, default=50, help="neighbours kept per movie in each similarity")
ap.add_argument("--shrink", type=float, default=10.0)
ap.add_argument("--like", type=float, default=4.0)
ap.add_argument("--content", default="models/sim_hybrid.npy")
args = ap.parse_args()

movies = pd.read_pickle("movies.pkl")
n_items = len(movies)
train = pd.read_csv("data/cf_train.csv")
test = pd.read_csv("data/cf_test.csv")

users = np.sort(train["user"].unique())
uidx = {u: i for i, u in enumerate(users)}
tr_u = train["user"].map(uidx).to_numpy()
tr_i = train["item"].to_numpy()
liked_tr = (train["rating"].to_numpy() >= args.like).astype(np.float32)

# user x item matrices from train
seen = np.zeros((len(users), n_items), dtype=bool)
seen[tr_u, tr_i] = True
liked = np.zeros((len(users), n_items), dtype=np.float32)
liked[tr_u[liked_tr > 0], tr_i[liked_tr > 0]] = 1.0

# ---- item-item CF similarity (cosine on liked matrix, shrunk by co-rating count) ----
norm = np.sqrt((liked ** 2).sum(axis=0)) + 1e-9
cos = (liked.T @ liked) / np.outer(norm, norm)
co = liked.T @ liked
cf_sim = cos * (co / (co + args.shrink))
np.fill_diagonal(cf_sim, 0)


def sparsify(S, k):
    """Keep each row's top-k positive entries, scaled so the row max is 1."""
    S = np.maximum(S, 0).astype(np.float32)
    out = np.zeros_like(S)
    for i in range(S.shape[0]):
        top = np.argpartition(S[i], -k)[-k:]
        out[i, top] = S[i, top]
        m = out[i].max()
        if m > 0:
            out[i] /= m
    return out


cf_sim = sparsify(cf_sim, args.neighbors)
content_sim = np.load(args.content).astype(np.float32)
np.fill_diagonal(content_sim, 0)
content_sim = sparsify(content_sim, args.neighbors)

cf_scores = liked @ cf_sim          # users x items
content_scores = liked @ content_sim
pop_scores = np.tile(seen.sum(axis=0).astype(np.float32), (len(users), 1))


def z(M):
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-9)


# ---- ground truth ----
rel = test[test["rating"] >= args.like]
rel_by_user = rel.groupby("user")["item"].apply(set).to_dict()
eval_users = [u for u in rel_by_user if u in uidx]
print(f"Users evaluated: {len(eval_users)} | train ratings: {len(train):,} | test ratings: {len(test):,} | movies: {n_items:,}")


def evaluate(scores):
    P, R, N, seen_items = [], [], [], set()
    disc = 1.0 / np.log2(np.arange(2, args.k + 2))
    for u in eval_users:
        r = uidx[u]
        s = scores[r].copy()
        s[seen[r]] = -np.inf
        top = np.argpartition(s, -args.k)[-args.k:]
        top = top[np.argsort(-s[top])]
        seen_items.update(top.tolist())
        truth = rel_by_user[u]
        hits = np.array([t in truth for t in top], dtype=float)
        P.append(hits.mean())
        R.append(hits.sum() / len(truth))
        ideal = disc[: min(len(truth), args.k)].sum()
        N.append((hits * disc).sum() / ideal)
    return {
        f"Precision@{args.k}": round(np.mean(P), 4),
        f"Recall@{args.k}": round(np.mean(R), 4),
        f"NDCG@{args.k}": round(np.mean(N), 4),
        "Coverage": round(len(seen_items) / n_items, 3),
    }


rows = [{"model": "popularity", **evaluate(pop_scores)},
        {"model": "content (hybrid)", **evaluate(content_scores)},
        {"model": "cf (item-item)", **evaluate(cf_scores)}]
zc, zt = z(cf_scores), z(content_scores)
for a in [0.25, 0.5, 0.75]:
    rows.append({"model": f"blend cf={a}", **evaluate(a * zc + (1 - a) * zt)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nHigher is better. Compare every row against 'popularity': a personalised model should beat it.")
