import os
import pickle
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "recommender", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "recommender", "data", "ml-1m")

def main():
    with open(os.path.join(MODEL_DIR, "knn.pkl"), "rb") as f:
        knn = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "popularity.pkl"), "rb") as f:
        pop = pickle.load(f)

    df = pd.read_csv(
        os.path.join(DATA_DIR, "ratings.dat"),
        sep="::", names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )

    all_movies = set(df["movie_id"].unique())
    user_ids = df["user_id"].unique()
    np.random.seed(42)
    sample_users = np.random.choice(user_ids, size=1000, replace=False)

    matrix = knn["matrix"]
    user_map = knn["user_map"]
    movie_ids = knn["movie_ids"]
    knn_recommended = set()

    for uid in sample_users:
        if uid not in user_map:
            continue
        u_idx = user_map[uid]
        user_vec = matrix[u_idx]
        user_sims = matrix @ user_vec.T
        scores = (matrix.T @ user_sims).toarray().flatten()
        for idx in user_vec.indices:
            scores[idx] = 0
        top = np.argsort(scores)[::-1][:20]
        for i in top:
            knn_recommended.add(int(movie_ids[i]))

    pop_recommended = set(pop["ranked"][:20])

    print("=== CATALOG COVERAGE ===")
    print(f"Total movies in catalog:         {len(all_movies)}")
    print(f"KNN unique movies recommended:   {len(knn_recommended)}")
    print(f"KNN catalog coverage:            {len(knn_recommended)/len(all_movies)*100:.1f}%")
    print(f"Popularity unique movies:        {len(pop_recommended)}")
    print(f"Popularity catalog coverage:     {len(pop_recommended)/len(all_movies)*100:.1f}%")

    print("\n=== EQUITY GAP ===")
    print("(from M3 subpopulation analysis)")
    print("KNN HR@20 light users:   0.3667")
    print("KNN HR@20 heavy users:   0.7906")
    print(f"Equity gap ratio:        {0.3667/0.7906:.4f}")
    print(f"Threshold (>=0.50):      {'PASS' if 0.3667/0.7906 >= 0.50 else 'FAIL'}")
    print("Pop HR@20 light users:   0.2848")
    print("Pop HR@20 heavy users:   0.5487")
    print(f"Equity gap ratio:        {0.2848/0.5487:.4f}")
    print(f"Threshold (>=0.50):      {'PASS' if 0.2848/0.5487 >= 0.50 else 'FAIL'}")

if __name__ == "__main__":
    main()
