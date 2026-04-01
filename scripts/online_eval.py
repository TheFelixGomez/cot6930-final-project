"""
Online Evaluator — scripts/online_eval.py
------------------------------------------
Joins gcl.reco_responses (recommended movie_ids per user) with
gcl.reco_requests (engagement events: view/click/purchase) to compute
a live Hit Rate @ K metric.

Proxy success definition:
    A recommendation is a "hit" if the user generated any engagement
    event (view, click, or purchase) on a recommended movie_id within
    WINDOW_MINUTES of the recommendation being served.

Run manually:   uv run python scripts/online_eval.py
Run in CI:      add as a scheduled GH Actions job (e.g. every 6 hours)

Dependencies:   confluent-kafka, python-decouple (already in pyproject.toml)
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from confluent_kafka import Consumer, KafkaError
from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL cert resolution — mirrors consumer.py pattern
# ---------------------------------------------------------------------------
_RENDER_CERT_PATH = "/etc/secrets/kafka-ca.pem"
CERT_DIR = "/etc/secrets" if os.path.exists(_RENDER_CERT_PATH) else "certs"

_SSL_CONF = {
    "security.protocol":        "SSL",
    "ssl.ca.location":          os.path.join(CERT_DIR, "kafka-ca.pem"),
    "ssl.certificate.location": os.path.join(CERT_DIR, "kafka-service.cert"),
    "ssl.key.location":         os.path.join(CERT_DIR, "kafka-service.key"),
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP   = config("KAFKA_BOOTSTRAP_SERVERS")
TEAM              = config("TEAM", default="gcl")
WINDOW_MINUTES    = config("EVAL_WINDOW_MINUTES", default=30, cast=int)
LOOKBACK_HOURS    = config("EVAL_LOOKBACK_HOURS", default=24, cast=int)
TOP_K             = config("EVAL_TOP_K", default=10, cast=int)

RESPONSES_TOPIC   = f"{TEAM}.reco_responses"   # probe responses (recommended ids)
REQUESTS_TOPIC    = f"{TEAM}.reco_requests"    # engagement events (view/click/purchase)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_consumer(group_id: str) -> Consumer:
    return Consumer({
        "bootstrap.servers":  KAFKA_BOOTSTRAP,
        "group.id":           group_id,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
        **_SSL_CONF,
    })


def _drain_topic(topic: str, group_id: str, max_empty_polls: int = 10) -> list[dict]:
    """Read all available messages from a topic and return as list of dicts."""
    consumer = _make_consumer(group_id)
    consumer.subscribe([topic])
    messages, empty_polls = [], 0

    try:
        while empty_polls < max_empty_polls:
            msg = consumer.poll(timeout=2.0)
            if msg is None:
                empty_polls += 1
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka error: %s", msg.error())
                empty_polls += 1
                continue
            try:
                messages.append(json.loads(msg.value()))
                empty_polls = 0
            except json.JSONDecodeError:
                pass
    finally:
        consumer.close()

    return messages


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def compute_hit_rate(
    responses: list[dict],
    engagements: list[dict],
    window_minutes: int,
    top_k: int,
    lookback_hours: int,
) -> dict:
    """
    For each recommendation response, check if the user engaged with
    any of the recommended movie_ids within window_minutes.

    Args:
        responses:      Records from gcl.reco_responses
        engagements:    Records from gcl.reco_requests (engagement events)
        window_minutes: Time window for a hit to count
        top_k:          Only consider first top_k recommended movies
        lookback_hours: Only evaluate records from the last N hours

    Returns:
        Dict with hit_rate, hits, total, and per-model breakdown
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    window = timedelta(minutes=window_minutes)

    # Build engagement lookup: user_id -> list of (movie_id, timestamp)
    engagement_index: dict[int, list[tuple[int, datetime]]] = defaultdict(list)
    for evt in engagements:
        try:
            payload = evt.get("payload", evt)
            uid     = int(payload.get("user_id", -1))
            mid     = int(payload.get("item_id", payload.get("movie_id", -1)))
            ts_raw  = evt.get("sent_at") or payload.get("ts")
            if uid < 0 or mid < 0 or not ts_raw:
                continue
            ts = _parse_iso(ts_raw) if isinstance(ts_raw, str) else \
                 datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            engagement_index[uid].append((mid, ts))
        except (ValueError, TypeError):
            continue

    hits, total = 0, 0
    model_stats: dict[str, dict] = defaultdict(lambda: {"hits": 0, "total": 0})

    for resp in responses:
        try:
            served_at = _parse_iso(resp.get("received_at", ""))
            if served_at < cutoff:
                continue

            body      = resp.get("response_body", {})
            user_id   = int(body.get("user_id", -1))
            model     = body.get("model_used", "unknown")
            movie_ids = [int(m) for m in body.get("movie_ids", [])[:top_k]]

            if user_id < 0 or not movie_ids:
                continue

            total += 1
            model_stats[model]["total"] += 1

            # Check if any recommended movie was engaged within the window
            user_engagements = engagement_index.get(user_id, [])
            is_hit = any(
                mid in movie_ids and served_at <= eng_ts <= served_at + window
                for mid, eng_ts in user_engagements
            )
            if is_hit:
                hits += 1
                model_stats[model]["hits"] += 1

        except (ValueError, TypeError, KeyError):
            continue

    hit_rate = hits / total if total > 0 else 0.0

    return {
        "hit_rate":      round(hit_rate, 4),
        "hits":          hits,
        "total":         total,
        "window_minutes": window_minutes,
        "lookback_hours": lookback_hours,
        "top_k":         top_k,
        "by_model": {
            m: {
                "hit_rate": round(s["hits"] / s["total"], 4) if s["total"] else 0.0,
                **s,
            }
            for m, s in model_stats.items()
        },
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    log.info("Draining %s...", RESPONSES_TOPIC)
    responses = _drain_topic(
        RESPONSES_TOPIC,
        group_id=f"{TEAM}_eval_responses",
    )
    log.info("Loaded %d response records", len(responses))

    log.info("Draining %s...", REQUESTS_TOPIC)
    engagements = _drain_topic(
        REQUESTS_TOPIC,
        group_id=f"{TEAM}_eval_requests",
    )
    log.info("Loaded %d engagement records", len(engagements))

    result = compute_hit_rate(
        responses=responses,
        engagements=engagements,
        window_minutes=WINDOW_MINUTES,
        top_k=TOP_K,
        lookback_hours=LOOKBACK_HOURS,
    )

    log.info("=== Online Evaluation Results ===")
    log.info("Lookback:   last %d hours", result["lookback_hours"])
    log.info("Window:     %d minutes", result["window_minutes"])
    log.info("Top-K:      %d", result["top_k"])
    log.info("Total recs: %d", result["total"])
    log.info("Hits:       %d", result["hits"])
    log.info("Hit Rate:   %.2f%%", result["hit_rate"] * 100)
    for model, stats in result["by_model"].items():
        log.info(
            "  [%s] hit_rate=%.2f%% (%d/%d)",
            model, stats["hit_rate"] * 100, stats["hits"], stats["total"],
        )


if __name__ == "__main__":
    run()
