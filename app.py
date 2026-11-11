import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_KEY = os.environ.get("TMDB_API_KEY")
PLACEHOLDER = "https://placehold.co/500x750?text=No+Image"


@st.cache_resource
def load_data():
    movies = pd.read_pickle("movies.pkl")
    nb = np.load("neighbors.npz")
    return movies, nb["idx"], nb["score"]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(movie_id):
    if not API_KEY:
        return PLACEHOLDER
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": API_KEY, "language": "en-US"},
            timeout=5,
        )
        r.raise_for_status()
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{path}" if path else PLACEHOLDER
    except requests.RequestException:
        return PLACEHOLDER


def recommend(title, n=5):
    i = movies.index[movies["title"] == title][0]
    top = neighbor_idx[i][:n]
    rows = movies.iloc[top]
    with ThreadPoolExecutor(max_workers=n) as ex:
        posters = list(ex.map(fetch_poster, rows["movie_id"].tolist()))
    return rows["title"].tolist(), posters, neighbor_score[i][:n]


movies, neighbor_idx, neighbor_score = load_data()

st.title("Movie Recommender System")
selected = st.selectbox("Select a movie", movies["title"].values)

if st.button("Recommend"):
    names, posters, scores = recommend(selected)
    for col, name, poster, sc in zip(st.columns(len(names)), names, posters, scores):
        with col:
            st.text(name)
            st.image(poster)
            st.caption(f"content match: {sc:.0%}")
