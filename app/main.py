import json
import logging
import pickle
import secrets
import uuid
import boto3

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from decouple import config
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from fastapi_limiter.depends import RateLimiter
from pyrate_limiter import Limiter, Rate, Duration

from app.kafka.consumer import start_consumer, stop_consumer
from app.recommender import recommend

log = logging.getLogger(__name__)
security = HTTPBasic()


def verify_metrics_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
):
    expected_user = config("METRICS_USER")
    expected_pass = config("METRICS_PASSWORD")

    correct_username = secrets.compare_digest(credentials.username, expected_user)
    correct_password = secrets.compare_digest(credentials.password, expected_pass)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


PIPELINE_GIT_SHA = config("PIPELINE_GIT_SHA", default="unknown")
CONTAINER_IMAGE_DIGEST = config("CONTAINER_IMAGE_DIGEST", default="unknown")

# R2 config for model registry hot-swap
R2_ACCOUNT_ID = config("R2_ACCOUNT_ID", default="")
R2_ACCESS_KEY = config("R2_ACCESS_KEY_ID", default="")
R2_SECRET = config("R2_SECRET_KEY", default="")
R2_BUCKET = config("R2_BUCKET", default="")

# In-memory model version state — updated by /switch
_model_state: dict = {
    "version": "local",
    "data_snapshot_id": "unknown",
    "trained_at": "unknown",
}


def _r2_client():
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET]):
        return None
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET,
        region_name="auto",
    )


def _load_version_from_registry(version: str) -> dict:
    """Download a versioned model from R2 and hot-swap it into the recommender."""

    import app.recommender as rec_module

    r2 = _r2_client()
    if r2 is None:
        raise RuntimeError("R2 credentials not configured")

    model_dir = Path("/tmp") / "model_registry" / version
    model_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("popularity.pkl", "knn.pkl", "metadata.json"):
        local = model_dir / fname
        if not local.exists():
            log.info("Downloading model_registry/%s/%s from R2...", version, fname)
            r2.download_file(R2_BUCKET, f"model_registry/{version}/{fname}", str(local))

    with open(model_dir / "popularity.pkl", "rb") as f:
        rec_module._popularity = pickle.load(f)
    with open(model_dir / "knn.pkl", "rb") as f:
        rec_module._knn = pickle.load(f)

    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)

    log.info("Hot-swapped to model version %s", version)
    return meta


def _resolve_latest_version() -> str:
    """Read model_registry/latest.json from R2 and return the version string."""
    r2 = _r2_client()
    if r2 is None:
        return "local"
    try:
        obj = r2.get_object(Bucket=R2_BUCKET, Key="model_registry/latest.json")
        return json.loads(obj["Body"].read()).get("version", "local")
    except Exception:
        return "local"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_consumer()

    # On startup, load the latest registered version if available
    latest = _resolve_latest_version()
    if latest != "local":
        try:
            meta = _load_version_from_registry(latest)
            _model_state.update(
                {
                    "version": meta.get("version", latest),
                    "data_snapshot_id": meta.get("data_snapshot_id", "unknown"),
                    "trained_at": meta.get("trained_at", "unknown"),
                }
            )
            log.info("Startup: loaded model version %s from registry", latest)
        except Exception as exc:
            log.warning(
                "Could not load registry version %s: %s — using local", latest, exc
            )
    else:
        log.info("Startup: using locally trained model (no registry version found)")

    yield
    stop_consumer()


app = FastAPI(lifespan=lifespan)


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="MovieLens integer user ID")
    n: int = Field(20, ge=1, le=100, description="Number of recommendations")
    model: Literal["knn", "popularity"] = Field("knn", description="Model to use")


class RecommendResponse(BaseModel):
    # Original fields — unchanged
    user_id: int
    model_used: str
    movie_ids: list[int]
    # Provenance fields — added for Task 5
    request_id: str
    model_version: str
    data_snapshot_id: str
    pipeline_git_sha: str


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.head("/ping")
async def ping_head():
    return {"message": "pong"}


@app.get("/version")
async def version():
    """Return current model version and provenance metadata."""

    return {
        "model_version": _model_state["version"],
        "data_snapshot_id": _model_state["data_snapshot_id"],
        "trained_at": _model_state["trained_at"],
        "pipeline_git_sha": PIPELINE_GIT_SHA,
        # container_image_digest kept internal - not exposed
    }


@app.post("/switch")
async def switch_model(
    model: str = Query(..., description="Version to load, e.g. v1.2 or 'latest'"),
):
    """Hot-swap the active model to a registered version without restarting."""

    target = _resolve_latest_version() if model == "latest" else model
    try:
        meta = _load_version_from_registry(target)
        _model_state.update(
            {
                "version": meta.get("version", target),
                "data_snapshot_id": meta.get("data_snapshot_id", "unknown"),
                "trained_at": meta.get("trained_at", "unknown"),
            }
        )
        return {
            "status": "switched",
            "model_version": _model_state["version"],
            "trained_at": _model_state["trained_at"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Switch failed: {exc}")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_movies(req: RecommendRequest):
    """Return top-n movie recommendations for a given user."""

    try:
        result = recommend(user_id=req.user_id, k=req.n, model=req.model)
    except RuntimeError as exc:
        log.error("Model artifacts unavailable for /recommend: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Recommendation service unavailable: model artifacts are missing.",
        )
    except Exception:
        log.exception("Unexpected error in /recommend for user_id=%s", req.user_id)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )

    return {
        **result,
        "request_id": str(uuid.uuid4()),
        "model_version": _model_state["version"],
        "data_snapshot_id": _model_state["data_snapshot_id"],
        "pipeline_git_sha": PIPELINE_GIT_SHA,
        # container_image_digest kept internal - not exposed
    }


# Set up Prometheus instrumentation for the app with credentials and rate limiting on the /metrics endpoint 1k/min
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    dependencies=[
        Depends(verify_metrics_credentials),
        Depends(RateLimiter(limiter=Limiter(Rate(1000, Duration.MINUTE)))),
    ],
)
