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

# Configurations

_RENDER_CERT_PATH = "/etc/secrets/kafka-ca.pem"
CERT_DIR = "/etc/secrets" if os.path.exists(_RENDER_CERT_PATH) else "certs"

_SSL_CONF = {
    "security.protocol":        "SSL",
    "ssl.ca.location":          os.path.join(CERT_DIR, "kafka-ca.pem"),
    "ssl.certificate.location": os.path.join(CERT_DIR, "kafka-service.cert"),
    "ssl.key.location":         os.path.join(CERT_DIR, "kafka-service.key"),
}

KAFKA_BOOTSTRAP   = config("KAFKA_BOOTSTRAP_SERVERS")
RECOMMEND_URL     = config("RECOMMEND_URL", default="")
API_KEY           = config("API_KEY",           default="")
REQUEST_TIMEOUT_S = config("REQUEST_TIMEOUT_S", default=60, cast=int)
TEAM              = config("TEAM", default="gcl")

REQUESTS_TOPIC  = f"{TEAM}.reco_requests"
RESPONSES_TOPIC = f"{TEAM}.reco_responses"


PROBE_PAYLOADS = [
    {"user_id": 1,     "n": 10, "model": "knn"},
    {"user_id": 1,     "n": 5,  "model": "popularity"},
    {"user_id": 99999, "n": 10, "model": "knn"},
]


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
        resp        = requests.post(
            RECOMMEND_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S
        )
        received_at = datetime.now(timezone.utc).isoformat()

        # Parse response body once — used for both logging and provenance
        body = resp.json() if resp.ok else {"error": resp.text[:512]}

        resp_record = {
            "probe_id":      probe_id,
            "team":          TEAM,
            "received_at":   received_at,
            "status_code":   resp.status_code,
            "latency_ms":    int(resp.elapsed.total_seconds() * 1000),
            "success":       resp.ok,
            "response_body": body,
            # Provenance fields — populated from /recommend response when present
            "request_id":             body.get("request_id",             probe_id),
            "model_version":          body.get("model_version",          "unknown"),
            "data_snapshot_id":       body.get("data_snapshot_id",       "unknown"),
            "pipeline_git_sha":       body.get("pipeline_git_sha",       "unknown"),
            "container_image_digest": body.get("container_image_digest", "unknown"),
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
            # Provenance fields — unknown on failure
            "request_id":             probe_id,
            "model_version":          "unknown",
            "data_snapshot_id":       "unknown",
            "pipeline_git_sha":       "unknown",
            "container_image_digest": "unknown",
        }
        log.error("Request failed (probe_id=%s): %s", probe_id, exc)

    return req_record, resp_record

def run() -> None:
    if not RECOMMEND_URL:
        log.error("RECOMMEND_URL is not set — skipping probe run.")
        sys.exit(1)

    producer = _producer()
    failures = 0

    for payload in PROBE_PAYLOADS:
        log.info("Probing %s | payload=%s", RECOMMEND_URL, payload)
        req_record, resp_record = call_recommend(payload)

        _produce(producer, REQUESTS_TOPIC,  req_record)
        _produce(producer, RESPONSES_TOPIC, resp_record)

        log.info(
            "probe_id=%s  status=%s  latency=%sms  success=%s  model_version=%s",
            req_record["probe_id"],
            resp_record["status_code"],
            resp_record["latency_ms"],
            resp_record["success"],
            resp_record["model_version"],
        )

        if not resp_record["success"]:
            failures += 1

    producer.flush()
    log.info("Probe run complete — %d/%d failed", failures, len(PROBE_PAYLOADS))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    run()