import json
import os
from confluent_kafka import Consumer
from decouple import config

KAFKA_BOOTSTRAP = config("KAFKA_BOOTSTRAP_SERVERS")
CERT_DIR = "certs"

def create_consumer(group_id):
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "security.protocol": "SSL",
        "ssl.ca.location": os.path.join(CERT_DIR, "kafka-ca.pem"),
        "ssl.certificate.location": os.path.join(CERT_DIR, "kafka-service.cert"),
        "ssl.key.location": os.path.join(CERT_DIR, "kafka-service.key"),
    })

def drain_topic(topic, group_id="availability_calc"):
    consumer = create_consumer(group_id)
    consumer.subscribe([topic])
    messages = []
    empty_count = 0

    while empty_count < 3:
        msg = consumer.poll(5.0)
        if msg is None:
            empty_count += 1
            continue
        if msg.error():
            empty_count += 1
            continue
        empty_count = 0
        try:
            data = json.loads(msg.value().decode("utf-8"))
            messages.append(data)
        except json.JSONDecodeError:
            continue

    consumer.close()
    return messages

def main():
    print("Draining gcl.reco_responses...")
    responses = drain_topic("gcl.reco_responses")
    print(f"Loaded {len(responses)} total probe responses\n")

    if not responses:
        print("No probe data available.")
        return

    success = sum(1 for r in responses if r.get("success", False))
    failed = len(responses) - success
    total = len(responses)
    availability = (success / total) * 100

    timestamps = []
    for r in responses:
        ts = r.get("received_at") or r.get("sent_at")
        if ts:
            timestamps.append(ts)

    if timestamps:
        timestamps.sort()
        print(f"First probe: {timestamps[0]}")
        print(f"Last probe:  {timestamps[-1]}")

    print(f"\n=== AVAILABILITY ===")
    print(f"Total probes:      {total}")
    print(f"Successful:        {success}")
    print(f"Failed:            {failed}")
    print(f"Availability:      {availability:.2f}%")
    print(f"Requirement:       >=70%")
    print(f"Met:               {'Yes' if availability >= 70 else 'No'}")

if __name__ == "__main__":
    main()