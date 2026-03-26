"""
Recommender — app/recommender.py
----------------------------------
Loads the trained popularity and KNN models from the models/ directory
and exposes a single recommend() function for the /recommend route.

Falls back to popularity ranking when:
  - the requested model is "knn" but the user_id is unknown
  - the KNN model returns an empty list

Models are loaded once at import time so the first request isn't slow.
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading — resolves to recommender/models/ relative to the repo root.
# train.py uses os.path.dirname(__file__) inside recommender/, so models land
# at recommender/models/. This file is at app/recommender.py, so we go:
#   app/recommender.py → parent = app/ → parent = repo root → recommender/models/
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent / "recommender" / "models"


def _load(name: str):
    path = _MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Run train.py first to generate the model files."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    _popularity = _load("popularity")
    _knn        = _load("knn")
    log.info("Loaded popularity and KNN models from %s", _MODEL_DIR)
except FileNotFoundError as exc:
    log.error("%s", exc)
    _popularity = None
    _knn        = None

# ---------------------------------------------------------------------------
# Inference helpers — mirror evaluate.py logic exactly
# ---------------------------------------------------------------------------

def _recommend_popularity(k: int) -> list[int]:
    if _popularity is None:
        return []
    return [int(m) for m in _popularity["ranked"][:k]]


def _recommend_knn(user_id: int, k: int) -> list[int]:
    if _knn is None:
        return []

    user_map   = _knn["user_map"]
    movie_ids  = _knn["movie_ids"]
    matrix     = _knn["matrix"]

    if user_id not in user_map:
        return []

    u_idx      = user_map[user_id]
    user_vec   = matrix[u_idx]
    rated      = set(user_vec.indices)

    user_sims  = matrix @ user_vec.T
    scores     = (matrix.T @ user_sims).toarray().flatten()

    for idx in rated:
        scores[idx] = 0

    top_indices = np.argsort(scores)[::-1][:k]
    return [int(movie_ids[i]) for i in top_indices]

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def recommend(user_id: int, k: int = 20, model: str = "knn") -> dict:
    """Return top-k movie recommendations for a user.

    Args:
        user_id: Integer user ID from the MovieLens dataset.
        k:       Number of recommendations to return (default 20).
        model:   "knn" (default) or "popularity".

    Returns:
        Dict with keys:
            user_id      - echoed back
            model_used   - which model actually served the request
            movie_ids    - list of recommended movie IDs (ints)
    """
    if model == "popularity" or _knn is None:
        return {
            "user_id":   user_id,
            "model_used": "popularity",
            "movie_ids": _recommend_popularity(k),
        }

    results = _recommend_knn(user_id, k)

    if not results:
        log.info("user_id=%s unknown to KNN — falling back to popularity", user_id)
        return {
            "user_id":    user_id,
            "model_used": "popularity",
            "movie_ids":  _recommend_popularity(k),
        }

    return {
        "user_id":    user_id,
        "model_used": "knn",
        "movie_ids":  results,
    }
