"""
Step 1 of collaborative filtering: link MovieLens ratings to your TMDB movie list and split them.

    python cf_prepare.py --ml-dir data/movielens

Needs: movies.pkl (movie_id = TMDB id), and MovieLens links.csv + ratings.csv in --ml-dir.
Writes: data/cf_train.csv, data/cf_test.csv with columns user, item, rating, timestamp
        where `item` is the row position of the movie in movies.pkl (same index the app uses).

Split: for every user, the latest 20% of their ratings (by time) go to test, the rest to train.
Users with fewer than --min-ratings matched ratings are dropped.
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ml-dir", default="data/movielens")
ap.add_argument("--min-ratings", type=int, default=10)
ap.add_argument("--test-frac", type=float, default=0.2)
args = ap.parse_args()

movies = pd.read_pickle("movies.pkl")
links = pd.read_csv(os.path.join(args.ml_dir, "links.csv")).dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)
ratings = pd.read_csv(os.path.join(args.ml_dir, "ratings.csv"))

# TMDB id -> row position in movies.pkl
pos = pd.DataFrame({"tmdbId": movies["movie_id"].astype(int).values, "item": np.arange(len(movies))})
pos = pos.drop_duplicates("tmdbId")

m = ratings.merge(links[["movieId", "tmdbId"]], on="movieId").merge(pos, on="tmdbId")
m = m.rename(columns={"userId": "user"})[["user", "item", "rating", "timestamp"]]

# keep users with enough matched ratings
counts = m.groupby("user").size()
m = m[m["user"].isin(counts[counts >= args.min_ratings].index)].copy()

# per-user temporal split
m = m.sort_values(["user", "timestamp"])
rank = m.groupby("user").cumcount()
size = m.groupby("user")["user"].transform("size")
is_test = rank >= (size * (1 - args.test_frac)).round().astype(int)
train, test = m[~is_test], m[is_test]

# drop test rows whose movie never appears in train (cannot be scored)
test = test[test["item"].isin(train["item"].unique())]

os.makedirs("data", exist_ok=True)
train.to_csv("data/cf_train.csv", index=False)
test.to_csv("data/cf_test.csv", index=False)

n_items = m["item"].nunique()
print(f"Ratings file: {len(ratings):,} ratings, {ratings['userId'].nunique():,} users")
print(f"Movies in your catalogue ({len(movies):,}) that have MovieLens ratings: {n_items:,} ({n_items / len(movies):.0%})")
print(f"After matching + filtering: {len(m):,} ratings, {m['user'].nunique():,} users")
print(f"Train: {len(train):,}   Test: {len(test):,}")
density = len(train) / (train['user'].nunique() * train['item'].nunique())
print(f"Train matrix density: {density:.2%}")
