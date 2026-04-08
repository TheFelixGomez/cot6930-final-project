import logging
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from decouple import config

from app.kafka.consumer import start_consumer, stop_consumer
from app.recommender import recommend

log = logging.getLogger(__name__)
security = HTTPBasic()


def verify_metrics_credentials(credentials: HTTPBasicCredentials = Depends(security)):
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Boot up: Start the Kafka background thread
    start_consumer()
    yield
    # Shut down: Signal the thread to stop and clean up
    stop_consumer()


app = FastAPI(lifespan=lifespan)


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="MovieLens integer user ID")
    n: int = Field(20, ge=1, le=100, description="Number of recommendations")
    model: Literal["knn", "popularity"] = Field("knn", description="Model to use")


class RecommendResponse(BaseModel):
    user_id: int
    model_used: str
    movie_ids: list[int]


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_movies(req: RecommendRequest):
    """Return top-n movie recommendations for a given user.

    The request body must be a JSON object matching ``RecommendRequest`` with:

    * ``user_id``: MovieLens integer user ID.
    * ``n``: Number of recommendations to return (defaults to 20, must be between 1 and 100).
    * ``model``: Recommendation model to use (e.g. ``"knn"`` or ``"popularity"``).

    On success, returns a ``RecommendResponse`` containing:

    * ``user_id``: The requested user ID.
    * ``model_used``: The model actually used to generate recommendations.
    * ``movie_ids``: A list of recommended movie IDs.

    Validation errors in the request body will result in a 422 response generated
    by FastAPI. If model artifacts are unavailable the endpoint returns 503.
    Unexpected errors from the underlying recommender will result in a 500 response.
    """
    try:
        result = recommend(user_id=req.user_id, k=req.n, model=req.model)
    except RuntimeError as exc:
        log.error("Model artifacts unavailable for /recommend: %s", exc)
        raise HTTPException(status_code=503, detail="Recommendation service unavailable: model artifacts are missing.")
    except Exception:
        log.exception("Unexpected error in /recommend for user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")
    
    return result


# Set up Prometheus instrumentation for the app
Instrumentator().instrument(app).expose(app,
                                        endpoint="/metrics",
                                        dependencies=[Depends(verify_metrics_credentials)])
