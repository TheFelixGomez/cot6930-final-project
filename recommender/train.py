import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from collections import Counter
from config import DATA_DIR, MODEL_DIR, TEST_RATIO, KNN_NEIGHBORS

def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )
    return df

def split_chronological(df, test_ratio=TEST_RATIO):
    df = df.sort_values(["user_id", "timestamp"])
    train_list, test_list = [], []
    for _, group in df.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])
    return pd.concat(train_list), pd.concat(test_list)

def train_popularity(train_df):
    start = time.time()
    counts = Counter(train_df["movie_id"])
    ranked = [movie for movie, _ in counts.most_common()]
    train_time = time.time() - start
    return {"ranked": ranked, "train_time": train_time}

def train_knn(train_df):
    start = time.time()

    user_ids = train_df["user_id"].unique()
    movie_ids = train_df["movie_id"].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    rows = train_df["user_id"].map(user_map).values
    cols = train_df["movie_id"].map(movie_map).values
    vals = train_df["rating"].values.astype(np.float32)
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    model = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, metric="cosine", algorithm="brute")
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

def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

def main():
    df = load_ratings()
    train_df, test_df = split_chronological(df)

    print(f"Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")

    print("Training popularity model...")
    pop = train_popularity(train_df)
    save_model(pop, "popularity")

    print("Training KNN model...")
    knn = train_knn(train_df)
    save_model(knn, "knn")

    # save test set
    os.makedirs(MODEL_DIR, exist_ok=True)
    test_df.to_parquet(os.path.join(MODEL_DIR, "test.parquet"), index=False)

    print(f"Popularity train time: {pop['train_time']:.2f}s")
    print(f"KNN train time: {knn['train_time']:.2f}s")
    print("Models saved.")

if __name__ == "__main__":
    main()