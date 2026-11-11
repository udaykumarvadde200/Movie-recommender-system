"""
Rebuild the movie features + similarity matrices from the raw TMDB 5000 CSVs.

Usage:
    python build_model.py --movies tmdb_5000_movies.csv --credits tmdb_5000_credits.csv
    python build_model.py ... --embeddings      # also build sentence-embedding model

Outputs (in ./models and ./):
    movies.pkl                 columns: movie_id, title, genres, overview, tags
    models/sim_count.npy       CountVectorizer + cosine
    models/sim_tfidf.npy       TF-IDF + cosine
    models/sim_embed.npy       sentence embeddings + cosine (optional)
    similarity.pkl.gz          the matrix the app loads (default: tfidf)
"""
import argparse
import ast
import gzip
import os
import pickle

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()


def names(text, limit=None):
    """Parse a JSON-ish column into a list of 'name' values."""
    try:
        items = [d["name"] for d in ast.literal_eval(text)]
    except (ValueError, SyntaxError, TypeError):
        return []
    return items[:limit] if limit else items


def director(text):
    try:
        return [d["name"] for d in ast.literal_eval(text) if d["job"] == "Director"][:1]
    except (ValueError, SyntaxError, TypeError):
        return []


def squash(items):
    """'Sam Worthington' -> 'samworthington' (keep each NAME as one token)."""
    return [i.replace(" ", "").lower() for i in items]


def stem(text):
    return " ".join(ps.stem(w) for w in text.split())


def load(movies_csv, credits_csv):
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv).rename(columns={"movie_id": "id"}).drop(columns="title")
    df = movies.merge(credits, on="id")
    df = df[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].dropna(
        subset=["overview"]
    )
    df["genres_list"] = df["genres"].apply(names)
    df["keywords_list"] = df["keywords"].apply(names)
    df["cast_list"] = df["cast"].apply(lambda x: names(x, 3))
    df["director_list"] = df["crew"].apply(director)

    # IMPORTANT: words are joined with SPACES. Only multi-word names are squashed.
    def build_tags(r):
        parts = (
            r["overview"].lower().split()
            + squash(r["genres_list"]) * 2  # genres weighted x2
            + squash(r["keywords_list"])
            + squash(r["cast_list"])
            + squash(r["director_list"])
        )
        return stem(" ".join(parts))

    df["tags"] = df.apply(build_tags, axis=1)
    out = pd.DataFrame(
        {
            "movie_id": df["id"].values,
            "title": df["title"].values,
            "genres": df["genres_list"].values,
            "overview": df["overview"].values,
            "tags": df["tags"].values,
        }
    ).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", required=True)
    ap.add_argument("--credits", required=True)
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--app-model", default="tfidf", choices=["count", "tfidf", "embed"])
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    movies = load(args.movies, args.credits)
    movies.to_pickle("movies.pkl")
    print("movies:", movies.shape)

    sims = {}
    cv = CountVectorizer(max_features=5000, stop_words="english")
    sims["count"] = cosine_similarity(cv.fit_transform(movies["tags"]))

    tf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    sims["tfidf"] = cosine_similarity(tf.fit_transform(movies["tags"]))

    if args.embeddings:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        text = (movies["overview"] + ". " + movies["genres"].apply(" ".join)).tolist()
        emb = model.encode(text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        sims["embed"] = emb @ emb.T

    for k, m in sims.items():
        np.save(f"models/sim_{k}.npy", m.astype("float32"))
        print(f"saved sim_{k}  shape={m.shape}")

    with gzip.open("similarity.pkl.gz", "wb") as f:
        pickle.dump(sims[args.app_model].astype("float32"), f)
    print("similarity.pkl.gz written from:", args.app_model)


if __name__ == "__main__":
    main()
