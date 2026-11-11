# Movie Recommender System

A content-based movie recommender built on the TMDB 5000 dataset. It compares three feature
approaches (bag-of-words, TF-IDF, sentence embeddings), blends the two best into a hybrid,
re-ranks by popularity, and serves recommendations through a Streamlit app.

> Screenshot: _add `screenshot.png` here_  
> Live demo: _add link after deploying_

## How it works

```
TMDB CSVs --> build_model.py --> similarity matrices (count, tfidf, embed)
                                        |
                                  hybrid.py  (blend count + embed)
                                        |
                                 finalize.py (popularity re-rank, top-20 neighbours)
                                        |
                                  neighbors.npz --> app.py (Streamlit)
```

1. **Features** (`build_model.py`): overview, genres (weighted x2), keywords, top-3 cast and director
   are combined into one tag string and stemmed. Multi-word names are kept as one token
   (e.g. `samworthington`).
2. **Three similarity models**, all cosine similarity:
   - `count`: CountVectorizer on the tags
   - `tfidf`: TF-IDF with unigrams and bigrams
   - `embed`: `all-MiniLM-L6-v2` sentence embeddings of overview + genres
3. **Hybrid** (`hybrid.py`): `alpha * z(count) + (1 - alpha) * z(embed)`, each matrix standardised per row.
   Chosen alpha: **0.5**.
4. **Popularity re-rank** (`finalize.py`): the top 50 candidates per movie are re-ordered with
   `0.8 * similarity + 0.2 * popularity`, where popularity is the percentile of the IMDB weighted rating.
   Only the top 20 neighbours per movie are stored (0.5 MB, versus 85 MB for the dense matrix).

## Results

Evaluated on 500 randomly sampled query movies, top-10 recommendations (`evaluate.py`, `hybrid.py`, `finalize.py`).

| Model | Precision@10 | Jaccard@10 | Coverage |
|---|---|---|---|
| count | 0.986 | 0.733 | 0.402 |
| tfidf | 0.935 | 0.723 | 0.545 |
| embed | 0.866 | 0.436 | 0.527 |
| hybrid (alpha 0.5) | 0.981 | 0.669 | 0.464 |

Effect of the popularity re-rank on the hybrid:

| Model | Precision@10 | Jaccard@10 | Coverage | Avg. popularity |
|---|---|---|---|---|
| hybrid | 0.981 | 0.669 | 0.464 | 0.486 |
| hybrid + re-rank (beta 0.2) | 0.980 | 0.669 | 0.462 | 0.561 |

- Precision@10: share of recommendations sharing at least one genre with the query movie.
- Jaccard@10: mean genre-set overlap, a stricter version of the same idea.
- Coverage: share of the catalogue that appears in at least one top-10 list.
- Avg. popularity: mean percentile of the IMDB weighted rating of the recommended movies.

**Qualitative check.** Embeddings gave the best sci-fi matches for *Avatar* (Star Trek Beyond, Serenity)
but were unreliable on movies with thin plot text, while count handled genre and cast well.
The hybrid kept both strengths. The re-rank removed obscure noise titles (for example for *Toy Story*)
without changing the genre metrics.

## Limitations

- **The metric is partly circular.** Genres are part of the tags that `count` and `tfidf` are built from,
  so genre-based scores favour them. Embeddings score lower on it but look better on some movies by eye.
  An independent evaluation needs real user ratings (planned, see below).
- Content-based only: no user history, no personalisation, and the dataset (about 4,800 movies) stops around 2017.
- Some weak matches remain (for example *Titanic* returns *Raise the Titanic*).

## Run it

```bash
pip install -r requirements-build.txt

# Download "TMDB 5000 Movie Dataset" from Kaggle into data/
python build_model.py --movies data/tmdb_5000_movies.csv --credits data/tmdb_5000_credits.csv --embeddings
python hybrid.py --alpha 0.5
python finalize.py --csv data/tmdb_5000_movies.csv
python -m streamlit run app.py
```

Optional: set `TMDB_API_KEY` to show posters. Without it, the app shows placeholder images.

```bash
export TMDB_API_KEY=your_key        # PowerShell: $env:TMDB_API_KEY="your_key"
```

`evaluate.py` prints the model comparison table and `compare.py` prints side-by-side recommendations for a few movies.

## Project structure

```
app.py             Streamlit app (loads movies.pkl + neighbors.npz)
build_model.py     Features and similarity matrices
hybrid.py          Blend count + embed similarity
finalize.py        Popularity re-rank and top-20 neighbour file
evaluate.py        Genre-based comparison of models
compare.py         Side-by-side recommendations for sample movies
movies.pkl         Movie metadata (id, title, genres, overview, tags)
neighbors.npz      Precomputed top-20 neighbours per movie
```

## Planned

- MovieLens collaborative filtering and a hybrid with the content model
- Evaluation on held-out user ratings instead of genre overlap

## Data

TMDB 5000 Movie Dataset (Kaggle). Posters and metadata via the TMDB API.
This product uses the TMDB API but is not endorsed or certified by TMDB.
