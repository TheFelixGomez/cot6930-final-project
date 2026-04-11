import json
import os
from scipy import stats
import numpy as np
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

def drain_topic(topic, group_id="ab_analysis_v2"):
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
    print(f"Loaded {len(responses)} responses\n")

    if not responses:
        print("No data to analyze.")
        return

    knn_latencies = []
    pop_latencies = []

    for r in responses:
        if not r.get("success", False):
            continue
        latency = r.get("latency_ms")
        if latency is None:
            continue

        body = r.get("response_body", {})
        model = body.get("model_used", "")
        if not model:
            payload = r.get("payload", {})
            model = payload.get("model", "")

        if model == "knn":
            knn_latencies.append(latency)
        elif model == "popularity":
            pop_latencies.append(latency)

    print(f"KNN responses:        {len(knn_latencies)}")
    print(f"Popularity responses: {len(pop_latencies)}")

    if not knn_latencies or not pop_latencies:
        print("Need responses from both models to run A/B test.")
        return

    knn_mean = np.mean(knn_latencies)
    pop_mean = np.mean(pop_latencies)
    knn_std = np.std(knn_latencies, ddof=1)
    pop_std = np.std(pop_latencies, ddof=1)

    # two-sample t-test (Welch's, does not assume equal variance)
    t_stat, p_value = stats.ttest_ind(knn_latencies, pop_latencies, equal_var=False)

    print(f"\n=== A/B TEST RESULTS (Latency Comparison) ===")
    print(f"{'Metric':<30} {'KNN':<15} {'Popularity':<15}")
    print("-" * 60)
    print(f"{'N':<30} {len(knn_latencies):<15} {len(pop_latencies):<15}")
    print(f"{'Mean Latency (ms)':<30} {knn_mean:<15.2f} {pop_mean:<15.2f}")
    print(f"{'Std Dev (ms)':<30} {knn_std:<15.2f} {pop_std:<15.2f}")
    print(f"{'Min (ms)':<30} {min(knn_latencies):<15} {min(pop_latencies):<15}")
    print(f"{'Max (ms)':<30} {max(knn_latencies):<15} {max(pop_latencies):<15}")
    print(f"\n{'T-statistic':<30} {t_stat:.4f}")
    print(f"{'P-value':<30} {p_value:.4f}")
    print(f"{'Significant (p < 0.05)':<30} {'Yes' if p_value < 0.05 else 'No'}")

    if p_value < 0.05:
        faster = "KNN" if knn_mean < pop_mean else "Popularity"
        slower = "Popularity" if faster == "KNN" else "KNN"
        print(f"\nDecision: {faster} has significantly lower latency than {slower}. Consider promoting {faster} as the default if accuracy is comparable.")
    else:
        print(f"\nDecision: No significant latency difference detected between models.")

if __name__ == "__main__":
    main()