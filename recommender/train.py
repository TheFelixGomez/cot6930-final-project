import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-1m")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )
    return df

def train_popularity(train_df):
    start = time.time()
    counts = Counter(train_df["movie_id"])
    ranked = [movie for movie, _ in counts.most_common()]
    train_time = time.time() - start
    return {"ranked": ranked, "train_time": train_time}

def train_knn(train_df):
    start = time.time()

    # map user and movie ids to matrix indices
    user_ids = train_df["user_id"].unique()
    movie_ids = train_df["movie_id"].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    # build user-item sparse matrix
    rows = train_df["user_id"].map(user_map).values
    cols = train_df["movie_id"].map(movie_map).values
    vals = train_df["rating"].values.astype(np.float32)
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    # fit item-based KNN on the transposed matrix (items as rows)
    model = NearestNeighbors(n_neighbors=21, metric="cosine", algorithm="brute")
    model.fit(matrix.T)

    train_time = time.time() - start

    return {
        "model": model,
        "matrix": matrix,
        "user_map": user_map,
        "movie_map": movie_map,
        "movie_ids": movie_ids,
        "train_time": train_time
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_ratings()

    # chronological split
    df = df.sort_values(["user_id", "timestamp"])
    train_list, test_list = [], []
    for _, group in df.groupby("user_id"):
        n_test = max(1, int(len(group) * 0.2))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    print(f"Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")

    print("Training popularity model...")
    pop = train_popularity(train_df)

    print("Training KNN model...")
    knn = train_knn(train_df)

    with open(os.path.join(MODEL_DIR, "popularity.pkl"), "wb") as f:
        pickle.dump(pop, f)

    with open(os.path.join(MODEL_DIR, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)

    test_df.to_parquet(os.path.join(MODEL_DIR, "test.parquet"), index=False)

    print(f"Popularity train time: {pop['train_time']:.2f}s")
    print(f"KNN train time: {knn['train_time']:.2f}s")
    print("Models saved.")

if __name__ == "__main__":
    main()