import numpy as np, pandas as pd, glob, os

movies = pd.read_pickle("movies.pkl")
tests = ["Avatar", "The Dark Knight", "Toy Story", "Titanic"]

for path in sorted(glob.glob("models/sim_*.npy")):
    sim = np.load(path)
    print("\n=====", os.path.basename(path)[4:-4], "=====")
    for t in tests:
        i = movies.index[movies["title"] == t][0]
        s = sim[i].copy(); s[i] = -1
        top = np.argsort(s)[::-1][:6]
        print(f"{t}: ", ", ".join(movies.title[j] for j in top))