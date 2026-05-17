import streamlit as st
import pickle
import gzip
import pandas as pd
import requests

API_KEY = "d9046203f374ef683edc6a3c7b9e7ebd"

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"

        response = requests.get(url, timeout=10)
        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"

    except Exception as e:
        print("Poster error:", e)
        return "https://via.placeholder.com/500x750?text=No+Image"


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        row = movies.iloc[i[0]]

        if "movie_id" in movies.columns:
            movie_id = row["movie_id"]
        elif "id" in movies.columns:
            movie_id = row["id"]
        else:
            st.error("No movie_id or id column found in movies.pkl")
            return [], []

        recommended_movies.append(row["title"])
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


movies_dict = pickle.load(open("movies.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

with gzip.open("similarity.pkl.gz", "rb") as f:
    similarity = pickle.load(f)


st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    if len(names) == 5:
        cols = st.columns(5)

        for index, col in enumerate(cols):
            with col:
                st.text(names[index])
                st.image(posters[index])
