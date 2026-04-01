"""
Quality Gates — tests/test_quality_gates.py
--------------------------------------------
Covers four quality gate categories:
    1. Unit tests        — ingestor validator, recommender fallback, /recommend route
    2. Schema validation — pandera checks on ingestor Parquet snapshots
    3. Drift detection   — user/movie distribution shift against training reference
    4. Backpressure      — ingestor flush retry and circuit breaker behaviour

Run:  uv run pytest tests/test_quality_gates.py -v
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from typing import Any

# ---------------------------------------------------------------------------
# Shared fixture: a valid probe record matching the fallback EVENT_SCHEMA
# (required: probe_id, team, sent_at, payload)
# ---------------------------------------------------------------------------
VALID_PROBE_RECORD = {
    "probe_id": "abc-123",
    "team":     "gcl",
    "sent_at":  "2026-03-26T22:21:29.347950+00:00",
    "payload":  {"user_id": 1, "n": 10, "model": "knn"},
}

# ---------------------------------------------------------------------------
# 1. UNIT TESTS
# ---------------------------------------------------------------------------

class TestIngestorValidator:
    """Validate that the ingestor schema checker accepts good records
    and rejects bad ones with an informative reason string."""

    def _validate(self, data: dict):
        from app.kafka.stream_ingestor import validate
        return validate(json.dumps(data).encode())

    def test_valid_record_passes(self):
        record, err = self._validate(VALID_PROBE_RECORD)
        assert err is None
        assert record["probe_id"] == "abc-123"

    def test_empty_bytes_rejected(self):
        from app.kafka.stream_ingestor import validate
        record, err = validate(b"")
        assert record is None
        assert "json_decode_error" in err

    def test_missing_probe_id_rejected(self):
        bad = {k: v for k, v in VALID_PROBE_RECORD.items() if k != "probe_id"}
        record, err = self._validate(bad)
        assert record is None
        assert "schema_validation_error" in err

    def test_missing_team_rejected(self):
        bad = {k: v for k, v in VALID_PROBE_RECORD.items() if k != "team"}
        record, err = self._validate(bad)
        assert record is None
        assert "schema_validation_error" in err

    def test_missing_sent_at_rejected(self):
        bad = {k: v for k, v in VALID_PROBE_RECORD.items() if k != "sent_at"}
        record, err = self._validate(bad)
        assert record is None
        assert "schema_validation_error" in err

    def test_missing_payload_rejected(self):
        bad = {k: v for k, v in VALID_PROBE_RECORD.items() if k != "payload"}
        record, err = self._validate(bad)
        assert record is None
        assert "schema_validation_error" in err

    def test_additional_properties_allowed(self):
        extra = {**VALID_PROBE_RECORD, "extra_field": "some_value"}
        record, err = self._validate(extra)
        assert err is None
        assert record["extra_field"] == "some_value"

    def test_invalid_json_rejected(self):
        from app.kafka.stream_ingestor import validate
        record, err = validate(b"{not valid json}")
        assert record pandase
        assert "json_decode_error" in err


class TestRecommenderFallback:
    """Verify the recommender falls back to popularity for unknown users."""

    def _build_mock_knn(self):
        user_ids  = np.array([1, 2, 3])
        movie_ids = np.array([100, 200, 300, 400])
        user_map  = {u: i for i, u in enumerate(user_ids)}
        data = np.array([5.0, 3.0, 4.0, 2.0, 1.0, 5.0], dtype=np.float32)
        rows = np.array([0, 0, 1, 1, 2, 2])
        cols = np.array([0, 1, 2, 3, 0, 3])
        matrix = csr_matrix((data, (rows, cols)), shape=(3, 4))
        return {
            "user_map":   user_map,
            "movie_ids":  movie_ids,
            "matrix":     matrix,
            "model":      MagicMock(),
            "train_time": 0.1,
        }

    def _build_mock_popularity(self):
        return {"ranked": [100, 200, 300, 400], "train_time": 0.01}

    def test_known_user_gets_knn_results(self):
        from app import recommender as rec
        with patch.object(rec, "_knn", self._build_mock_knn()), \
             patch.object(rec, "_popularity", self._build_mock_popularity()):
            result = rec.recommend(user_id=1, k=2, model="knn")
        assert result["model_used"] == "knn"
        assert len(result["movie_ids"]) == 2

    def test_unknown_user_falls_back_to_popularity(self):
        from app import recommender as rec
        with patch.object(rec, "_knn", self._build_mock_knn()), \
             patch.object(rec, "_popularity", self._build_mock_popularity()):
            result = rec.recommend(user_id=99999, k=4, model="knn")
        assert result["model_used"] == "popularity"
        assert result["movie_ids"] == [100, 200, 300, 400]

    def test_explicit_popularity_model(self):
        from app import recommender as rec
        with patch.object(rec, "_knn", self._build_mock_knn()), \
             patch.object(rec, "_popularity", self._build_mock_popularity()):
            result = rec.recommend(user_id=1, k=2, model="popularity")
        assert result["model_used"] == "popularity"

    def test_none_knn_falls_back(self):
        from app import recommender as rec
        with patch.object(rec, "_knn", None), \
             patch.object(rec, "_popularity", self._build_mock_popularity()):
            result = rec.recommend(user_id=1, k=4, model="knn")
        assert result["model_used"] == "popularity"


class TestRecommendRoute:
    """FastAPI route tests — validate request shapes and response schema."""

    @pytest.fixture
    def client(self):
        from app.main import app
        with patch("app.kafka.consumer.start_consumer"), \
             patch("app.kafka.consumer.stop_consumer"), \
             patch("app.recommender.recommend") as mock_rec:
            mock_rec.return_value = {
                "user_id":    1,
                "model_used": "knn",
                "movie_ids":  [100, 200, 300],
            }
            with TestClient(app) as c:
                yield c

    def test_valid_request_returns_200(self, client):
        resp = client.post("/recommend", json={"user_id": 1, "n": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert "movie_ids"   in body
        assert "model_used"  in body
        assert "user_id"     in body

    def test_missing_user_id_returns_422(self, client):
        resp = client.post("/recommend", json={"n": 5})
        assert resp.status_code == 422

    def test_n_out_of_range_returns_422(self, client):
        assert client.post("/recommend", json={"user_id": 1, "n": 0}).status_code   == 422
        assert client.post("/recommend", json={"user_id": 1, "n": 101}).status_code == 422

    def test_invalid_model_returns_422(self, client):
        resp = client.post("/recommend", json={"user_id": 1, "model": "gbm"})
        assert resp.status_code == 422

    def test_ping_route(self, client):
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json() == {"message": "pong"}


# ---------------------------------------------------------------------------
# 2. SCHEMA VALIDATION (pandera)
# Validates the shape of DataFrames the ingestor writes to R2.
# Columns reflect the probe record fields after flattening.
# ---------------------------------------------------------------------------

SNAPSHOT_SCHEMA = pa.DataFrameSchema(
    columns={
        "probe_id": pa.Column(str, nullable=False),
        "team":     pa.Column(str, nullable=False),
        "sent_at":  pa.Column(str, nullable=False),
        "payload":  pa.Column(object, nullable=False),
    },
    strict=False,  # allow extra columns
)


class TestSchemaValidation:
    """Validate ingestor snapshot DataFrames conform to the expected schema."""

    def _make_df(self, overrides: dict | None = None) -> pd.DataFrame:
        row = {**VALID_PROBE_RECORD}
        if overrides:
            row.update(overrides)
        return pd.DataFrame([row])

    def test_valid_dataframe_passes(self):
        SNAPSHOT_SCHEMA.validate(self._make_df())

    def test_null_probe_id_fails(self):
        with pytest.raises(pa.errors.SchemaError):
            SNAPSHOT_SCHEMA.validate(self._make_df({"probe_id": None}))

    def test_null_team_fails(self):
        with pytest.raises(pa.errors.SchemaError):
            SNAPSHOT_SCHEMA.validate(self._make_df({"team": None}))

    def test_null_sent_at_fails(self):
        with pytest.raises(pa.errors.SchemaError):
            SNAPSHOT_SCHEMA.validate(self._make_df({"sent_at": None}))

    def test_null_payload_fails(self):
        with pytest.raises(pa.errors.SchemaError):
            SNAPSHOT_SCHEMA.validate(self._make_df({"payload": None}))

    def test_extra_columns_allowed(self):
        df = self._make_df({"extra_field": "bonus"})
        SNAPSHOT_SCHEMA.validate(df)

    def test_multiple_rows_pass(self):
        rows = [
            {**VALID_PROBE_RECORD, "probe_id": f"id-{i}"}
            for i in range(5)
        ]
        SNAPSHOT_SCHEMA.validate(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# 3. DRIFT DETECTION
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """
    Detect distribution shift by comparing user_id / movie_id coverage
    in a new batch against the known training population.
    """

    KNOWN_USERS  = {1, 2, 3, 4, 5}
    KNOWN_MOVIES = {100, 200, 300, 400, 500}

    def _compute_drift(
        self,
        batch: pd.DataFrame,
        known_users:  set,
        known_movies: set,
        user_threshold:  float = 0.3,
        movie_threshold: float = 0.3,
    ) -> dict:
        unknown_users  = batch["user_id"].apply(
            lambda u: int(u) not in known_users
        ).mean()
        unknown_movies = batch["movie_id"].apply(
            lambda m: int(m) not in known_movies
        ).mean()
        return {
            "unknown_user_rate":   round(float(unknown_users),  4),
            "unknown_movie_rate":  round(float(unknown_movies), 4),
            "user_drift_flagged":  unknown_users  > user_threshold,
            "movie_drift_flagged": unknown_movies > movie_threshold,
        }

    def _make_batch(self, user_ids: list, movie_ids: list) -> pd.DataFrame:
        return pd.DataFrame({
            "user_id":  [str(u) for u in user_ids],
            "movie_id": [str(m) for m in movie_ids],
        })

    def test_clean_batch_no_drift(self):
        batch  = self._make_batch([1, 2, 3], [100, 200, 300])
        report = self._compute_drift(batch, self.KNOWN_USERS, self.KNOWN_MOVIES)
        assert not report["user_drift_flagged"]
        assert not report["movie_drift_flagged"]

    def test_high_unknown_users_flags_drift(self):
        batch  = self._make_batch([999, 888, 777], [100, 200, 300])
        report = self._compute_drift(batch, self.KNOWN_USERS, self.KNOWN_MOVIES)
        assert report["user_drift_flagged"]

    def test_high_unknown_movies_flags_drift(self):
        batch  = self._make_batch([1, 2, 3], [9001, 9002, 9003])
        report = self._compute_drift(batch, self.KNOWN_USERS, self.KNOWN_MOVIES)
        assert report["movie_drift_flagged"]

    def test_mixed_batch_below_threshold(self):
        # 1 unknown out of 5 = 20% < 30% threshold — should not flag
        batch  = self._make_batch([1, 2, 3, 4, 999], [100, 200, 300, 400, 500])
        report = self._compute_drift(batch, self.KNOWN_USERS, self.KNOWN_MOVIES)
        assert not report["user_drift_flagged"]

    def test_drift_rate_accuracy(self):
        # 2 unknown out of 4 = exactly 50%
        batch  = self._make_batch([1, 2, 999, 888], [100, 200, 300, 400])
        report = self._compute_drift(batch, self.KNOWN_USERS, self.KNOWN_MOVIES)
        assert report["unknown_user_rate"] == 0.5

    def test_no_drift_below_custom_threshold(self):
        # 40% unknown — below a custom 50% threshold
        batch  = self._make_batch([1, 2, 3, 999, 888], [100, 200, 300, 400, 500])
        report = self._compute_drift(
            batch, self.KNOWN_USERS, self.KNOWN_MOVIES,
            user_threshold=0.5,
        )
        assert not report["user_drift_flagged"]


# ---------------------------------------------------------------------------
# 4. BACKPRESSURE HANDLING
# flush_batch(batch, r2_client) — r2_client is passed in, not module-level
# ---------------------------------------------------------------------------

class TestBackpressure:
    """Verify flush_batch retries on transient S3/R2 errors and raises
    after MAX_FLUSH_RETRIES are exhausted."""

    def _make_batch(self) -> list[dict[str, Any]]:
        return [dict(VALID_PROBE_RECORD)]

    def _mock_r2(self):
        return MagicMock()

    def test_flush_succeeds_on_first_try(self):
        from app.kafka.stream_ingestor import flush_batch
        r2 = self._mock_r2()
        r2.put_object.return_value = {}
        with patch("app.kafka.stream_ingestor.redis", None):
            flush_batch(self._make_batch(), r2)
        assert r2.put_object.call_count == 1

    def test_flush_retries_on_transient_error(self):
        import botocore.exceptions as bce
        from app.kafka.stream_ingestor import flush_batch
        error = bce.ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "slow"}},
            "PutObject",
        )
        r2 = self._mock_r2()
        r2.put_object.side_effect = [error, error, {}]
        with patch("app.kafka.stream_ingestor.redis", None), \
             patch("app.kafka.stream_ingestor.time") as mock_time, \
             patch("app.kafka.stream_ingestor.MAX_FLUSH_RETRIES", 3):
            mock_time.sleep = MagicMock()
            flush_batch(self._make_batch(), r2)
        assert r2.put_object.call_count == 3

    def test_flush_raises_after_max_retries(self):
        import botocore.exceptions as bce
        from app.kafka.stream_ingestor import flush_batch
        error = bce.ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "down"}},
            "PutObject",
        )
        r2 = self._mock_r2()
        r2.put_object.side_effect = error
        with patch("app.kafka.stream_ingestor.redis", None), \
             patch("app.kafka.stream_ingestor.time") as mock_time, \
             patch("app.kafka.stream_ingestor.MAX_FLUSH_RETRIES", 2):
            mock_time.sleep = MagicMock()
            with pytest.raises(Exception):
                flush_batch(self._make_batch(), r2)
        assert r2.put_object.call_count == 2

    def test_empty_batch_skips_flush(self):
        from app.kafka.stream_ingestor import flush_batch
        r2 = self._mock_r2()
        flush_batch([], r2)
        r2.put_object.assert_not_called()

    def test_redis_cache_updated_on_success(self):
        from app.kafka.stream_ingestor import flush_batch
        r2    = self._mock_r2()
        redis = MagicMock()
        r2.put_object.return_value = {}
        with patch("app.kafka.stream_ingestor.redis", redis):
            flush_batch(self._make_batch(), r2)
        redis.set.assert_called_once()
