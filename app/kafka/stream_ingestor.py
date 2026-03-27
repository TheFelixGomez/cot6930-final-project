"""
Stream Ingestor — app/kafka/ingestor.py
----------------------------------------
Consumes from gcl.reco_requests, validates messages against
schemas/event.json, batches valid records, and writes Parquet/CSV
snapshots to object storage (S3 or Cloudflare R2).

Invalid messages are forwarded to gcl.reco_requests.dlq.

Mirrors the SSL config pattern from app/kafka/consumer.py.
Run standalone: uv run python -m app.kafka.ingestor

Dependencies (uv add):
    pandas pyarrow boto3 jsonschema redis
"""

import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import jsonschema
import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer
from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL cert resolution — matches consumer.py pattern exactly
# ---------------------------------------------------------------------------
_RENDER_CERT_PATH = "/etc/secrets/kafka-ca.pem"
CERT_DIR = "/etc/secrets" if os.path.exists(_RENDER_CERT_PATH) else "certs"

# ---------------------------------------------------------------------------
# Config — via python-decouple (.env or environment variables)
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP   = config("KAFKA_BOOTSTRAP_SERVERS")         # matches consumer.py
KAFKA_GROUP_ID    = config("KAFKA_INGESTOR_GROUP_ID",  default="gcl_ingestor_group")
KAFKA_INPUT_TOPIC = config("KAFKA_INPUT_TOPIC",        default="gcl.reco_requests")
KAFKA_DLQ_TOPIC   = config("KAFKA_DLQ_TOPIC",          default="gcl.reco_requests.dlq")

# Cloudflare R2 config
# R2_ACCOUNT_ID    → your Cloudflare account ID (found in R2 dashboard)
# R2_ACCESS_KEY_ID → R2 API token Access Key ID
# R2_SECRET_KEY    → R2 API token Secret Access Key
# R2_BUCKET        → bucket name
R2_ACCOUNT_ID     = config("R2_ACCOUNT_ID",    default=None)
R2_ACCESS_KEY_ID  = config("R2_ACCESS_KEY_ID", default=None)
R2_SECRET_KEY     = config("R2_SECRET_KEY",    default=None)
R2_BUCKET         = config("R2_BUCKET",        default=None)
R2_PREFIX         = config("R2_PREFIX",                default="snapshots/")
OUTPUT_FORMAT     = config("OUTPUT_FORMAT",            default="parquet")  # "parquet" | "csv"

BATCH_SIZE        = config("BATCH_SIZE",               default=5,  cast=int)
FLUSH_INTERVAL_S  = config("FLUSH_INTERVAL_S",         default=10,   cast=int)

REDIS_URL         = config("REDIS_URL",                default="")
REDIS_TTL_S       = config("REDIS_TTL_S",              default=3600, cast=int)

# ---------------------------------------------------------------------------
# Shared Kafka SSL config — same cert paths as consumer.py
# ---------------------------------------------------------------------------
_SSL_CONF = {
    "security.protocol":      "SSL",
    "ssl.ca.location":        os.path.join(CERT_DIR, "kafka-ca.pem"),
    "ssl.certificate.location": os.path.join(CERT_DIR, "kafka-service.cert"),
    "ssl.key.location":       os.path.join(CERT_DIR, "kafka-service.key"),
}

# ---------------------------------------------------------------------------
# Schema — loaded from shared schemas/event.json
# ---------------------------------------------------------------------------
_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "event.json"

try:
    with _SCHEMA_PATH.open() as f:
        EVENT_SCHEMA = json.load(f)
    log.info("Loaded schema from %s", _SCHEMA_PATH)
except FileNotFoundError:
    log.warning(
        "schemas/event.json not found — using minimal fallback schema. "
        "Commit the agreed schema before deploying."
    )
    EVENT_SCHEMA = {
        "type": "object",
        "required": ["user_id", "item_id", "event_type", "ts"],
        "properties": {
            "user_id":    {"type": "string"},
            "item_id":    {"type": "string"},
            "event_type": {"type": "string", "enum": ["view", "click", "purchase"]},
            "ts":         {"type": "number"},
        },
        "additionalProperties": True,
    }

_validator = jsonschema.Draft7Validator(EVENT_SCHEMA)

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def _make_r2():
    """Create a boto3 client pointed at Cloudflare R2.

    R2 exposes an S3-compatible API at:
        https://<account_id>.r2.cloudflarestorage.com
    boto3 talks to it identically to S3 — only the endpoint and
    credentials differ.
    """
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",  # R2 requires "auto" — not a real AWS region
    )


def _make_redis():
    if not REDIS_URL:
        return None
    import redis as redis_lib
    return redis_lib.from_url(REDIS_URL)


r2    = _make_r2()
redis = _make_redis()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate(raw: bytes) -> tuple[dict | None, str | None]:
    """Parse and validate a raw Kafka message value.

    Returns (record, None) on success, (None, error_reason) on failure.
    """
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"

    errors = list(_validator.iter_errors(record))
    if errors:
        reason = "; ".join(e.message for e in errors)
        return None, f"schema_validation_error: {reason}"

    return record, None


def flush_batch(batch: list[dict[str, Any]]) -> None:
    """Serialize a batch to Parquet or CSV and upload to object storage."""
    if not batch:
        return

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df  = pd.DataFrame(batch)
    buf = io.BytesIO()

    if OUTPUT_FORMAT == "parquet":
        df.to_parquet(buf, index=False, engine="pyarrow")
        ext, content_type = "parquet", "application/octet-stream"
    else:
        df.to_csv(buf, index=False)
        ext, content_type = "csv", "text/csv"

    buf.seek(0)
    key = f"{R2_PREFIX}{ts}.{ext}"

    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=buf.getvalue(), ContentType=content_type)
    log.info("Flushed %d records → r2://%s/%s", len(batch), R2_BUCKET, key)

    if redis:
        cache_key = f"ingestor:latest_snapshot:{KAFKA_INPUT_TOPIC}"
        redis.set(cache_key, f"r2://{R2_BUCKET}/{key}", ex=REDIS_TTL_S)
        log.debug("Redis cache updated: %s", cache_key)


def send_to_dlq(producer: Producer, raw: bytes, reason: str) -> None:
    producer.produce(
        KAFKA_DLQ_TOPIC,
        value=raw,
        headers={"error": reason.encode()},
    )
    producer.poll(0)

# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def run() -> None:
    consumer = Consumer({
        "bootstrap.servers":  KAFKA_BOOTSTRAP,
        "group.id":           KAFKA_GROUP_ID,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
        **_SSL_CONF,
    })
    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        **_SSL_CONF,
    })

    consumer.subscribe([KAFKA_INPUT_TOPIC])
    log.info("Ingestor subscribed to %s (group: %s)", KAFKA_INPUT_TOPIC, KAFKA_GROUP_ID)

    batch: list[dict] = []
    last_flush = time.monotonic()

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                pass
            elif msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka error: %s", msg.error())
            else:
                record, err = validate(msg.value())
                if err:
                    log.warning("Invalid msg offset=%s: %s", msg.offset(), err)
                    send_to_dlq(producer, msg.value(), err)
                else:
                    batch.append(record)
                consumer.commit(message=msg, asynchronous=False)

            elapsed = time.monotonic() - last_flush
            if len(batch) >= BATCH_SIZE or (batch and elapsed >= FLUSH_INTERVAL_S):
                flush_batch(batch)
                batch = []
                last_flush = time.monotonic()

    except KeyboardInterrupt:
        log.info("Shutdown — flushing %d remaining records", len(batch))
        flush_batch(batch)
    finally:
        producer.flush()
        consumer.close()


if __name__ == "__main__":
    run()
