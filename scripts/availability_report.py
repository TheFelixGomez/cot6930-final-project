import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from confluent_kafka import Consumer
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
TEAM              = config("TEAM", default="gcl")
RESPONSES_TOPIC   = f"{TEAM}.reco_responses"

HOURS_BEFORE      = config("AVAILABILITY_HOURS_BEFORE", default=72,  cast=int)
HOURS_AFTER       = config("AVAILABILITY_HOURS_AFTER",  default=144, cast=int)
SUBMISSION_TS_STR = config("SUBMISSION_TS", default="")

# Drain topic

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


def _compute_availability(
    records: list[dict],
    window_start: datetime,
    window_end: datetime,
    label: str,
) -> dict:
    """Filter records to window and compute success rate."""
    in_window = []
    for rec in records:
        try:
            ts = datetime.fromisoformat(
                rec.get("received_at", "")
            ).replace(tzinfo=timezone.utc)
            if window_start <= ts <= window_end:
                in_window.append(rec)
        except (ValueError, TypeError):
            continue

    total   = len(in_window)
    success = sum(1 for r in in_window if r.get("success") is True)
    failed  = total - success
    rate    = (success / total * 100) if total > 0 else 0.0

    # Group by hour for the timeline
    hourly: dict[str, dict] = defaultdict(lambda: {"success": 0, "failed": 0})
    for rec in in_window:
        try:
            ts  = datetime.fromisoformat(rec["received_at"]).replace(tzinfo=timezone.utc)
            key = ts.strftime("%Y-%m-%dT%H:00Z")
            if rec.get("success"):
                hourly[key]["success"] += 1
            else:
                hourly[key]["failed"] += 1
        except (ValueError, TypeError, KeyError):
            continue

    return {
        "label":        label,
        "window_start": window_start.isoformat(),
        "window_end":   window_end.isoformat(),
        "total_probes": total,
        "successful":   success,
        "failed":       failed,
        "availability": round(rate, 2),
        "meets_70pct":  rate >= 70.0,
        "hourly":       dict(sorted(hourly.items())),
    }


def _print_report(r: dict) -> None:
    bar_len  = 30
    filled   = int(r["availability"] / 100 * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)
    status   = "✓ PASS" if r["meets_70pct"] else "✗ FAIL"

    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"{'='*60}")
    print(f"  Window : {r['window_start']}")
    print(f"         → {r['window_end']}")
    print(f"  Probes : {r['total_probes']} total  "
          f"({r['successful']} success, {r['failed']} failed)")
    print(f"  Uptime : [{bar}] {r['availability']:.1f}%  {status}")
    print("  Req    : ≥70.0%")

    if r["hourly"]:
        print("\n  Hourly breakdown (success/failed):")
        for hour, counts in list(r["hourly"].items())[-24:]:  # last 24 hours
            s = counts["success"]
            f = counts["failed"]
            indicator = "✓" if f == 0 else ("~" if s > 0 else "✗")
            print(f"    {indicator} {hour}  {s} ok / {f} fail")


def run() -> None:
    now = datetime.now(timezone.utc)

    # Determine submission timestamp
    if SUBMISSION_TS_STR:
        try:
            submission_ts = datetime.fromisoformat(SUBMISSION_TS_STR).replace(tzinfo=timezone.utc)
        except ValueError:
            log.warning("Invalid SUBMISSION_TS format, using now as submission time")
            submission_ts = now
    else:
        submission_ts = now
        log.info("SUBMISSION_TS not set — using current time as submission point")

    log.info("Draining %s...", RESPONSES_TOPIC)
    records = _drain(RESPONSES_TOPIC, group_id=f"{TEAM}_availability_report")
    log.info("Loaded %d response records", len(records))

    # Pre-submission window: 72h before submission
    pre_start = submission_ts - timedelta(hours=HOURS_BEFORE)
    pre_end   = submission_ts

    # Post-submission window: 144h after submission
    post_start = submission_ts
    post_end   = submission_ts + timedelta(hours=HOURS_AFTER)

    pre  = _compute_availability(records, pre_start,  pre_end,   f"PRE-SUBMISSION  ({HOURS_BEFORE}h before)")
    post = _compute_availability(records, post_start, post_end,  f"POST-SUBMISSION ({HOURS_AFTER}h after)")

    _print_report(pre)
    _print_report(post)

    # Overall verdict
    print(f"\n{'='*60}")
    print("  OVERALL VERDICT")
    print(f"{'='*60}")
    print(f"  Pre-submission  availability : {pre['availability']:.1f}%  "
          f"{'✓' if pre['meets_70pct'] else '✗'}")
    print(f"  Post-submission availability : {post['availability']:.1f}%  "
          f"{'✓' if post['meets_70pct'] else '✗'}")
    both_pass = pre["meets_70pct"] and post["meets_70pct"]
    print(f"  Both windows ≥ 70%          : {'✓ PASS' if both_pass else '✗ FAIL — check failed probe runs'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
