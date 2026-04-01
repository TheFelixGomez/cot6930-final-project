import os
import time
import pickle
import numpy as np
import pandas as pd
from config import MODEL_DIR, EVAL_K, EVAL_SAMPLE_SIZE, RANDOM_SEED, DATA_DIR

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
    user_vec = matrix[u_idx]
    rated_indices = set(user_vec.indices)

    user_sims = matrix @ user_vec.T
    scores = (matrix.T @ user_sims).toarray().flatten()

    for idx in rated_indices:
        scores[idx] = 0

    top_indices = np.argsort(scores)[::-1][:k]
    return [movie_ids[i] for i in top_indices]

def hit_rate_at_k(recommend_fn, model, test_df, k=20):
    hits = 0
    total = 0
    for user_id, group in test_df.groupby("user_id"):
        actual = set(group["movie_id"])
        predicted = recommend_fn(model, user_id, k)
        if set(predicted) & actual:
            hits += 1
        total += 1
    return hits / total

def ndcg_at_k(recommend_fn, model, test_df, k=20):
    scores = []
    for user_id, group in test_df.groupby("user_id"):
        actual = set(group["movie_id"])
        predicted = recommend_fn(model, user_id, k)

        dcg = 0
        for i, mid in enumerate(predicted):
            if mid in actual:
                dcg += 1.0 / np.log2(i + 2)

        ideal_hits = min(len(actual), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        if idcg > 0:
            scores.append(dcg / idcg)
        else:
            scores.append(0)
    return np.mean(scores)

def avg_latency(recommend_fn, model, user_ids, k=20, n=100):
    sample = np.random.choice(user_ids, size=min(n, len(user_ids)), replace=False)
    start = time.time()
    for uid in sample:
        recommend_fn(model, uid, k)
    elapsed = time.time() - start
    return (elapsed / len(sample)) * 1000

def subpopulation_analysis(recommend_fn, model, train_df, test_df, k=20):
    """Split users into light/medium/heavy based on training set activity."""
    user_counts = train_df.groupby("user_id").size()

    terciles = user_counts.quantile([0.33, 0.66])
    low = terciles.iloc[0]
    high = terciles.iloc[1]

    light_users = set(user_counts[user_counts <= low].index)
    medium_users = set(user_counts[(user_counts > low) & (user_counts <= high)].index)
    heavy_users = set(user_counts[user_counts > high].index)

    results = {}
    for label, user_set in [("Light", light_users), ("Medium", medium_users), ("Heavy", heavy_users)]:
        subset = test_df[test_df["user_id"].isin(user_set)]
        if len(subset) == 0:
            continue
        hr = hit_rate_at_k(recommend_fn, model, subset, k)
        ndcg = ndcg_at_k(recommend_fn, model, subset, k)
        results[label] = {"HR@" + str(k): round(hr, 4), "NDCG@" + str(k): round(ndcg, 4), "Users": len(user_set)}

    return results

def main():
    k = EVAL_K
    np.random.seed(RANDOM_SEED)

    # load data
    test_df = pd.read_parquet(os.path.join(MODEL_DIR, "test.parquet"))
    train_df_path = os.path.join(DATA_DIR, "ratings.dat")
    train_full = pd.read_csv(
        train_df_path, sep="::", names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )
    # recreate train split to get user activity counts
    train_full = train_full.sort_values(["user_id", "timestamp"])
    train_list = []
    for _, group in train_full.groupby("user_id"):
        n_test = max(1, int(len(group) * 0.2))
        train_list.append(group.iloc[:-n_test])
    train_df = pd.concat(train_list)

    pop = load_model("popularity")
    knn = load_model("knn")
    user_ids = test_df["user_id"].unique()

    # sample for evaluation
    sample_users = np.random.choice(user_ids, size=EVAL_SAMPLE_SIZE, replace=False)
    sample_test = test_df[test_df["user_id"].isin(sample_users)]
    sample_train = train_df[train_df["user_id"].isin(sample_users)]

    print(f"Evaluating on {EVAL_SAMPLE_SIZE} sampled users...\n")

    # overall metrics
    print("=== OVERALL METRICS ===")
    pop_hr = hit_rate_at_k(recommend_popularity, pop, sample_test, k)
    knn_hr = hit_rate_at_k(recommend_knn, knn, sample_test, k)
    pop_ndcg = ndcg_at_k(recommend_popularity, pop, sample_test, k)
    knn_ndcg = ndcg_at_k(recommend_knn, knn, sample_test, k)
    pop_lat = avg_latency(recommend_popularity, pop, user_ids, k)
    knn_lat = avg_latency(recommend_knn, knn, user_ids, k)
    pop_size = os.path.getsize(os.path.join(MODEL_DIR, "popularity.pkl")) / (1024 * 1024)
    knn_size = os.path.getsize(os.path.join(MODEL_DIR, "knn.pkl")) / (1024 * 1024)

    print(f"\n{'Metric':<25} {'Popularity':<15} {'KNN (Item-CF)':<15}")
    print("-" * 55)
    print(f"{'HR@' + str(k):<25} {pop_hr:<15.4f} {knn_hr:<15.4f}")
    print(f"{'NDCG@' + str(k):<25} {pop_ndcg:<15.4f} {knn_ndcg:<15.4f}")
    print(f"{'Train Time (s)':<25} {pop['train_time']:<15.2f} {knn['train_time']:<15.2f}")
    print(f"{'Inference Latency (ms)':<25} {pop_lat:<15.3f} {knn_lat:<15.3f}")
    print(f"{'Model Size (MB)':<25} {pop_size:<15.2f} {knn_size:<15.2f}")

    # subpopulation analysis
    print("\n\n=== SUBPOPULATION ANALYSIS (KNN) ===")
    print("Users split into terciles by training set rating count.\n")
    knn_subpop = subpopulation_analysis(recommend_knn, knn, sample_train, sample_test, k)

    print(f"{'Group':<10} {'Users':<10} {'HR@' + str(k):<12} {'NDCG@' + str(k):<12}")
    print("-" * 44)
    for group, metrics in knn_subpop.items():
        print(f"{group:<10} {metrics['Users']:<10} {metrics['HR@' + str(k)]:<12} {metrics['NDCG@' + str(k)]:<12}")

    print("\n\n=== SUBPOPULATION ANALYSIS (Popularity) ===\n")
    pop_subpop = subpopulation_analysis(recommend_popularity, pop, sample_train, sample_test, k)

    print(f"{'Group':<10} {'Users':<10} {'HR@' + str(k):<12} {'NDCG@' + str(k):<12}")
    print("-" * 44)
    for group, metrics in pop_subpop.items():
        print(f"{group:<10} {metrics['Users']:<10} {metrics['HR@' + str(k)]:<12} {metrics['NDCG@' + str(k)]:<12}")

if __name__ == "__main__":
    main()