import os
import time
import pickle
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model(name):
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

def recommend_popularity(model, user_id, k=20):
    return model["ranked"][:k]

def recommend_knn(model, user_id, k=20):
    user_map = model["user_map"]
    movie_ids = model["movie_ids"]
    matrix = model["matrix"]

    if user_id not in user_map:
        return []

    u_idx = user_map[user_id]
    user_vec = matrix[u_idx]  # (1, n_movies)
    rated_indices = set(user_vec.indices)

    # find similar users, then aggregate their ratings into item scores
    user_sims = matrix @ user_vec.T  # (n_users, 1)
    scores = (matrix.T @ user_sims).toarray().flatten()  # (n_movies,)

    for idx in rated_indices:
        scores[idx] = 0

    top_indices = np.argsort(scores)[::-1][:k]
    return [movie_ids[i] for i in top_indices]

def hit_rate(recommend_fn, model, test_df, k=20):
    hits = 0
    total = 0
    for user_id, group in test_df.groupby("user_id"):
        actual = set(group["movie_id"])
        predicted = recommend_fn(model, user_id, k)
        if set(predicted) & actual:
            hits += 1
        total += 1
    return hits / total

def avg_latency(recommend_fn, model, user_ids, k=20, n=100):
    sample = np.random.choice(user_ids, size=min(n, len(user_ids)), replace=False)
    start = time.time()
    for uid in sample:
        recommend_fn(model, uid, k)
    elapsed = time.time() - start
    return (elapsed / len(sample)) * 1000

def main():
    test_df = pd.read_parquet(os.path.join(MODEL_DIR, "test.parquet"))
    pop = load_model("popularity")
    knn = load_model("knn")
    user_ids = test_df["user_id"].unique()
    k = 20

    np.random.seed(42)
    sample_users = np.random.choice(user_ids, size=1000, replace=False)
    sample_test = test_df[test_df["user_id"].isin(sample_users)]

    print("Evaluating on 1000 sampled users...")

    pop_hr = hit_rate(recommend_popularity, pop, sample_test, k)
    knn_hr = hit_rate(recommend_knn, knn, sample_test, k)

    pop_lat = avg_latency(recommend_popularity, pop, user_ids, k)
    knn_lat = avg_latency(recommend_knn, knn, user_ids, k)

    pop_size = os.path.getsize(os.path.join(MODEL_DIR, "popularity.pkl")) / (1024 * 1024)
    knn_size = os.path.getsize(os.path.join(MODEL_DIR, "knn.pkl")) / (1024 * 1024)

    print(f"\n{'Metric':<25} {'Popularity':<15} {'KNN (Item-CF)':<15}")
    print("-" * 55)
    print(f"{'HR@' + str(k):<25} {pop_hr:<15.4f} {knn_hr:<15.4f}")
    print(f"{'Train Time (s)':<25} {pop['train_time']:<15.2f} {knn['train_time']:<15.2f}")
    print(f"{'Inference Latency (ms)':<25} {pop_lat:<15.3f} {knn_lat:<15.3f}")
    print(f"{'Model Size (MB)':<25} {pop_size:<15.2f} {knn_size:<15.2f}")

if __name__ == "__main__":
    main()