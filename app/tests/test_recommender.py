from unittest.mock import patch, mock_open
import pytest
import numpy as np
import scipy.sparse as sp
from app import recommender

import importlib


# ---------------------------------------------------------------------------
# Test loading logic (_load function)
# ---------------------------------------------------------------------------

def test_load_file_not_found():
    # Force Path.exists() to return False
    with patch("app.recommender.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            recommender._load("fake_model")


def test_load_success():
    # Force Path.exists() to return True and mock the file reading
    with patch("app.recommender.Path.exists", return_value=True):
        with patch("app.recommender.open", mock_open(read_data=b"dummy")):
            with patch("app.recommender.pickle.load", return_value={"status": "loaded"}):
                result = recommender._load("fake_model")
                assert result == {"status": "loaded"}


# ---------------------------------------------------------------------------
# Test internal recommendation helpers
# ---------------------------------------------------------------------------

@patch("app.recommender._popularity", None)
def test_recommend_popularity_none():
    assert recommender._recommend_popularity(10) == []


@patch("app.recommender._knn", None)
def test_recommend_knn_none():
    assert recommender._recommend_knn(1, 10) == []


# ---------------------------------------------------------------------------
# Test the public recommend() interface
# ---------------------------------------------------------------------------

@patch("app.recommender._popularity", None)
@patch("app.recommender._knn", None)
def test_recommend_no_models_available():
    with pytest.raises(RuntimeError, match="No model artifacts are available"):
        recommender.recommend(user_id=1)


@patch("app.recommender._popularity", {"ranked": [101, 102, 103, 104, 105]})
@patch("app.recommender._knn", "dummy_knn_exists_but_wont_be_used")
def test_recommend_explicit_popularity():
    # Test requesting popularity directly
    result = recommender.recommend(user_id=1, k=3, model="popularity")
    assert result["model_used"] == "popularity"
    assert result["movie_ids"] == [101, 102, 103]


@patch("app.recommender._popularity", {"ranked": [101, 102, 103]})
@patch("app.recommender._knn", {"user_map": {}, "movie_ids": [], "matrix": None})
def test_recommend_knn_fallback_unknown_user():
    # Test fallback: user not in the KNN user_map
    result = recommender.recommend(user_id=999, k=2, model="knn")
    assert result["model_used"] == "popularity"
    assert result["movie_ids"] == [101, 102]


def test_recommend_knn_success():
    # We need to simulate the KNN matrix math.
    # Create a tiny 2x3 user-item sparse matrix using SciPy (CSR format like the real model)
    # User 0 rated Item 0
    # User 1 rated Item 0 and Item 1
    data = np.array([1, 1, 1])
    row_ind = np.array([0, 1, 1])
    col_ind = np.array([0, 0, 1])
    mock_matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=(2, 3))
    
    mock_knn = {
        "user_map": {100: 0, 200: 1},  # User ID 100 is at matrix index 0
        "movie_ids": [1000, 1001, 1002],  # Matrix column indices map to these Movie IDs
        "matrix": mock_matrix
    }
    
    with patch("app.recommender._knn", mock_knn):
        # User 100 wants recommendations.
        # They rated movie 1000 (idx 0).
        # User 200 is similar and rated movie 1001 (idx 1).
        # It should recommend movie 1001!
        result = recommender.recommend(user_id=100, k=1, model="knn")
        
        assert result["model_used"] == "knn"
        assert result["movie_ids"] == [1001]


# ---------------------------------------------------------------------------
# Test module-level import logic (the try/except block at the top of the file)
# ---------------------------------------------------------------------------

def test_module_import_failure():
    # We patch pathlib's Path directly so reload() cannot overwrite our mock.
    # This forces the FileNotFoundError to cover the 'except' block.
    with patch("pathlib.Path.exists", return_value=False):
        importlib.reload(recommender)
        assert recommender._popularity is None
        assert recommender._knn is None
