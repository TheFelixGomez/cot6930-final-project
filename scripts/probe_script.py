"""
Recommendation API Probe — scripts/probe.py
--------------------------------------------
Fires sample payloads at the /recommend endpoint and writes
request + response records to Kafka topics:
    gcl.reco_requests
    gcl.reco_responses

Uses the same SSL cert resolution pattern as app/kafka/consumer.py.
Exit code 1 on any probe failure — GH Actions marks the job failed.

Dependencies:
    requests
    (confluent-kafka and python-decouple already in pyproject.toml)
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import requests
from confluent_kafka import Producer
from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL cert resolution — mirrors app/kafka/consumer.py exactly
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
KAFKA_BOOTSTRAP   = config("KAFKA_BOOTSTRAP_SERVERS")       # matches consumer.py
RECOMMEND_URL     = config("RECOMMEND_URL")
API_KEY           = config("API_KEY",           default="")
REQUEST_TIMEOUT_S = config("REQUEST_TIMEOUT_S", default=10, cast=int)

# Team is hardcoded to gcl based on consumer.py — override via env if needed
TEAM = config("TEAM", default="gcl")

REQUESTS_TOPIC  = f"{TEAM}.reco_requests"
RESPONSES_TOPIC = f"{TEAM}.reco_responses"

# ---------------------------------------------------------------------------
# Probe payloads
# NOTE: update these fields to match the actual /recommend request schema
# once that route is added to app/main.py
# ---------------------------------------------------------------------------
PROBE_PAYLOADS = [
    {"user_id": 1,   "n": 10, "model": "knn"},
    {"user_id": 1,   "n": 5, "model": "popularity"},
    {"user_id": 99999,   "n": 10, "model": "knn"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        **_SSL_CONF,
    })


def _produce(producer: Producer, topic: str, payload: dict) -> None:
    producer.produce(
        topic,
        key=payload.get("probe_id", str(uuid.uuid4())).encode(),
        value=json.dumps(payload).encode(),
    )
    producer.poll(0)


def call_recommend(payload: dict) -> tuple[dict, dict]:
    """POST to /recommend and return (request_record, response_record)."""
    probe_id = str(uuid.uuid4())
    sent_at  = datetime.now(timezone.utc).isoformat()

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req_record = {
        "probe_id": probe_id,
        "team":     TEAM,
        "sent_at":  sent_at,
        "url":      RECOMMEND_URL,
        "payload":  payload,
    }

    try:
        resp        = requests.post(RECOMMEND_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S)
        received_at = datetime.now(timezone.utc).isoformat()
        resp_record = {
            "probe_id":      probe_id,
            "team":          TEAM,
            "received_at":   received_at,
            "status_code":   resp.status_code,
            "latency_ms":    int(resp.elapsed.total_seconds() * 1000),
            "success":       resp.ok,
            "response_body": resp.json() if resp.ok else {"error": resp.text[:512]},
        }
    except requests.exceptions.RequestException as exc:
        received_at = datetime.now(timezone.utc).isoformat()
        resp_record = {
            "probe_id":      probe_id,
            "team":          TEAM,
            "received_at":   received_at,
            "status_code":   None,
            "latency_ms":    None,
            "success":       False,
            "response_body": {"error": str(exc)},
        }
        log.error("Request failed (probe_id=%s): %s", probe_id, exc)

    return req_record, resp_record

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    producer = _producer()
    failures = 0

    for payload in PROBE_PAYLOADS:
        log.info("Probing %s | payload=%s", RECOMMEND_URL, payload)
        req_record, resp_record = call_recommend(payload)

        _produce(producer, REQUESTS_TOPIC,  req_record)
        _produce(producer, RESPONSES_TOPIC, resp_record)

        log.info(
            "probe_id=%s  status=%s  latency=%sms  success=%s",
            req_record["probe_id"],
            resp_record["status_code"],
            resp_record["latency_ms"],
            resp_record["success"],
        )

        if not resp_record["success"]:
            failures += 1

    producer.flush()
    log.info("Probe run complete — %d/%d failed", failures, len(PROBE_PAYLOADS))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    run()
