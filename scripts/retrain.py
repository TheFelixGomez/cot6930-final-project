"""
Automated Retraining + Model Registry — scripts/retrain.py
------------------------------------------------------------
1. Downloads latest MovieLens data (skips if already present)
2. Trains popularity and KNN models via recommender/train.py
3. Versions the artifacts as vMAJOR.MINOR (auto-incremented from R2)
4. Uploads models + metadata to R2 under model_registry/vX.Y/
5. Writes model_registry/latest.json pointing to the new version

Run manually:   uv run python scripts/retrain.py
Run in CI:      .github/workflows/retrain.yml (cron or workflow_dispatch)

Env vars required (same R2 credentials already in .env):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_KEY, R2_BUCKET
    PIPELINE_GIT_SHA  — injected by GH Actions via ${{ github.sha }}
    CONTAINER_IMAGE_DIGEST — optional, injected by docker/build-push-action
"""

import json
import logging
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Configuration

R2_ACCOUNT_ID  = config("R2_ACCOUNT_ID")
R2_ACCESS_KEY  = config("R2_ACCESS_KEY_ID")
R2_SECRET      = config("R2_SECRET_KEY")
R2_BUCKET      = config("R2_BUCKET")
GIT_SHA        = config("PIPELINE_GIT_SHA",       default="unknown")
IMAGE_DIGEST   = config("CONTAINER_IMAGE_DIGEST", default="unknown")

REPO_ROOT  = Path(__file__).parent.parent
MODEL_DIR  = REPO_ROOT / "recommender" / "models"
MAJOR      = 1   # bump manually for breaking changes


def _r2() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET,
        region_name="auto",
    )


def _next_version(r2_client) -> str:
    """Read latest.json from R2 to determine next minor version."""
    try:
        obj = r2_client.get_object(Bucket=R2_BUCKET, Key="model_registry/latest.json")
        latest = json.loads(obj["Body"].read())
        current = latest.get("version", f"v{MAJOR}.0")
        minor = int(current.split(".")[1]) + 1
    except r2_client.exceptions.NoSuchKey:
        minor = 1
    except Exception:
        minor = 1
    return f"v{MAJOR}.{minor}"


def _get_data_snapshot_id(r2_client) -> str:
    """Return the key of the most recent ingestor snapshot in R2."""
    try:
        resp = r2_client.list_objects_v2(
            Bucket=R2_BUCKET,
            Prefix="snapshots/",
        )
        objects = sorted(
            resp.get("Contents", []),
            key=lambda o: o["LastModified"],
            reverse=True,
        )
        return objects[0]["Key"] if objects else "unknown"
    except Exception:
        return "unknown"


def _run_training() -> None:
    log.info("Running download_data.py...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "recommender" / "download_data.py")],
        check=True,
    )
    log.info("Running train.py...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "recommender" / "train.py")],
        check=True,
    )


def _upload_artifact(r2_client, local_path: Path, r2_key: str) -> None:
    log.info("Uploading %s → s3://%s/%s", local_path.name, R2_BUCKET, r2_key)
    r2_client.upload_file(str(local_path), R2_BUCKET, r2_key)


def _upload_registry(r2_client, version: str, metadata: dict) -> None:
    """Upload model artifacts and metadata for a given version."""
    prefix = f"model_registry/{version}"

    # Upload model pickle files
    for fname in ("popularity.pkl", "knn.pkl"):
        local = MODEL_DIR / fname
        if not local.exists():
            raise FileNotFoundError(f"Expected model file not found: {local}")
        _upload_artifact(r2_client, local, f"{prefix}/{fname}")

    # Upload metadata
    meta_bytes = json.dumps(metadata, indent=2).encode()
    r2_client.put_object(
        Bucket=R2_BUCKET,
        Key=f"{prefix}/metadata.json",
        Body=meta_bytes,
        ContentType="application/json",
    )
    log.info("Uploaded metadata for %s", version)

    # Update latest.json
    r2_client.put_object(
        Bucket=R2_BUCKET,
        Key="model_registry/latest.json",
        Body=json.dumps({"version": version, "updated_at": metadata["trained_at"]}).encode(),
        ContentType="application/json",
    )
    log.info("Updated model_registry/latest.json → %s", version)


def run() -> None:
    r2_client = _r2()
    version   = _next_version(r2_client)
    log.info("Target version: %s", version)

    _run_training()

    # Compute training stats from saved artifacts
    with open(MODEL_DIR / "popularity.pkl", "rb") as f:
        pop = pickle.load(f)
    with open(MODEL_DIR / "knn.pkl", "rb") as f:
        knn = pickle.load(f)

    data_snapshot_id = _get_data_snapshot_id(r2_client)

    metadata = {
        "version":            version,
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "pipeline_git_sha":   GIT_SHA,
        "container_image_digest": IMAGE_DIGEST,
        "data_snapshot_id":   data_snapshot_id,
        "popularity_train_time_s": round(pop["train_time"], 3),
        "knn_train_time_s":   round(knn["train_time"], 3),
        "n_users":            len(knn["user_map"]),
        "n_movies":           len(knn["movie_ids"]),
    }

    _upload_registry(r2_client, version, metadata)

    log.info("=== Retraining complete ===")
    log.info("Version:          %s", version)
    log.info("Git SHA:          %s", GIT_SHA)
    log.info("Data snapshot:    %s", data_snapshot_id)
    log.info("Users in model:   %d", metadata["n_users"])
    log.info("Movies in model:  %d", metadata["n_movies"])


if __name__ == "__main__":
    run()
