import json
import logging
import os
from datetime import datetime, timezone, timedelta

from confluent_kafka import Consumer, KafkaError
from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


_RENDER_CERT_PATH = "/etc/secrets/kafka-ca.pem"
CERT_DIR = "/etc/secrets" if os.path.exists(_RENDER_CERT_PATH) else "certs"

_SSL_CONF = {
    "security.protocol":        "SSL",
    "ssl.ca.location":          os.path.join(CERT_DIR, "kafka-ca.pem"),
    "ssl.certificate.location": os.path.join(CERT_DIR, "kafka-service.cert"),
    "ssl.key.location":         os.path.join(CERT_DIR, "kafka-service.key"),
}

KAFKA_BOOTSTRAP  = config("KAFKA_BOOTSTRAP_SERVERS")
TEAM             = config("TEAM", default="gcl")
FILTER_ID        = config("REQUEST_ID", default="")
LOOKBACK_HOURS   = config("TRACE_LOOKBACK_HOURS", default=168, cast=int)
RESPONSES_TOPIC  = f"{TEAM}.reco_responses"


def _drain(topic: str, group_id: str, max_empty: int = 10) -> list[dict]:
    consumer = Consumer({
        "bootstrap.servers":  KAFKA_BOOTSTRAP,
        "group.id":           group_id,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
        **_SSL_CONF,
    })
    consumer.subscribe([topic])
    records, empty = [], 0
    try:
        while empty < max_empty:
            msg = consumer.poll(2.0)
            if msg is None:
                empty += 1
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka error: %s", msg.error())
                empty += 1
                continue
            try:
                records.append(json.loads(msg.value()))
                empty = 0
            except json.JSONDecodeError:
                pass
    finally:
        consumer.close()
    return records


PROVENANCE_FIELDS = [
    "request_id",
    "model_version",
    "data_snapshot_id",
    "pipeline_git_sha",
    "container_image_digest",
]

def _extract_provenance(resp: dict) -> dict | None:
    """Extract provenance fields from a reco_response record."""
    body = resp.get("response_body", {})

    # Support both flat and nested response shapes
    provenance = {
        "request_id":             resp.get("probe_id", body.get("request_id", "unknown")),
        "model_version":          body.get("model_version",          "unknown"),
        "data_snapshot_id":       body.get("data_snapshot_id",       "unknown"),
        "pipeline_git_sha":       body.get("pipeline_git_sha",       "unknown"),
        "container_image_digest": body.get("container_image_digest", "unknown"),
        # Extra context
        "user_id":      body.get("user_id",    "unknown"),
        "model_used":   body.get("model_used", "unknown"),
        "movie_ids":    body.get("movie_ids",  []),
        "received_at":  resp.get("received_at", "unknown"),
        "status_code":  resp.get("status_code", "unknown"),
        "latency_ms":   resp.get("latency_ms",  "unknown"),
    }

    # Filter by request_id if specified
    if FILTER_ID and provenance["request_id"] != FILTER_ID:
        return None

    return provenance


def _print_trace(p: dict, index: int) -> None:
    print(f"\n{'='*60}")
    print(f"  PREDICTION TRACE #{index}")
    print(f"{'='*60}")
    print(f"  request_id             : {p['request_id']}")
    print(f"  model_version          : {p['model_version']}")
    print(f"  data_snapshot_id       : {p['data_snapshot_id']}")
    print(f"  pipeline_git_sha       : {p['pipeline_git_sha']}")
    print(f"  container_image_digest : {p['container_image_digest']}")
    print("  ---")
    print(f"  user_id                : {p['user_id']}")
    print(f"  model_used             : {p['model_used']}")
    print(f"  movie_ids              : {p['movie_ids'][:5]}{'...' if len(p['movie_ids']) > 5 else ''}")
    print(f"  received_at            : {p['received_at']}")
    print(f"  status_code            : {p['status_code']}")
    print(f"  latency_ms             : {p['latency_ms']}")


def run() -> None:
    log.info("Draining %s...", RESPONSES_TOPIC)
    records = _drain(RESPONSES_TOPIC, group_id=f"{TEAM}_provenance_trace")
    log.info("Loaded %d response records", len(records))

    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    traces, skipped = [], 0

    for rec in records:
        try:
            received_at_str = rec.get("received_at", "")
            if received_at_str:
                received_at = datetime.fromisoformat(received_at_str).replace(tzinfo=timezone.utc)
                if received_at < cutoff:
                    skipped += 1
                    continue
        except (ValueError, TypeError):
            pass

        p = _extract_provenance(rec)
        if p:
            traces.append(p)

    log.info("Traces within lookback window: %d (skipped %d outside window)", len(traces), skipped)

    if not traces:
        log.warning("No traces found. Run the probe first or widen TRACE_LOOKBACK_HOURS.")
        return

    # Print most recent 5 traces
    for i, trace in enumerate(traces[-5:], start=1):
        _print_trace(trace, i)

    # Summary table
    versions = [t["model_version"] for t in traces]
    git_shas  = [t["pipeline_git_sha"][:8] for t in traces if t["pipeline_git_sha"] != "unknown"]
    print(f"\n{'='*60}")
    print(f"  PROVENANCE SUMMARY  ({len(traces)} predictions)")
    print(f"{'='*60}")
    print(f"  Model versions seen  : {sorted(set(versions))}")
    print(f"  Git SHAs seen        : {sorted(set(git_shas))}")
    print(f"  All fields present   : {all(t['request_id'] != 'unknown' for t in traces)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()