# Real-Time Movie Recommendation Service

## Project Overview
This repository contains a real-time movie recommendation service built with FastAPI, Kafka, and Docker. It serves predictions and processes real-time user engagement streams.

## Repository Structure (subject to change)
* `/app`: FastAPI application (`app.py`, routes, schema validation).
* `/app/kafka`: Kafka consumers and producers (`consumer.py`).
* `Dockerfile`: Container definition for the web service.
* `pyproject.toml`: Pinned Python dependencies.
* `.github/workflows`: CI/CD pipelines for testing and deployment.

## Development Setup (subject to change)
1. Clone the repository.
2. Install `uv` if not already installed: `pip install uv`
3. Create a virtual environment and install dependencies: `uv sync`
4. Copy `.env.example` to `.env` and fill in Kafka/API credentials.
5. Run the server locally: `uv run fastapi dev app/app.py`


## Kafka Consumer
The Kafka consumer runs in a background thread and is started/stopped by FastAPI's lifespan hooks in `app/main.py`.

### Lifecycle behavior
* On app startup, `start_consumer()` is called.
* `start_consumer()` creates a daemon thread that runs `consume_loop()`.
* On app shutdown, `stop_consumer()` sets the run flag to `False` and joins the thread for graceful shutdown.

### Consumer configuration
Current config in `app/kafka/consumer.py`:

* `bootstrap.servers`: from environment variable `KAFKA_BOOTSTRAP_SERVERS`
* `group.id`: `gcl_consumer_group`
* `auto.offset.reset`: `earliest`
* `security.protocol`: `SSL`
* `ssl.ca.location`: `certs/kafka-ca.pem`
* `ssl.certificate.location`: `certs/kafka-service.cert`
* `ssl.key.location`: `certs/kafka-service.key`
* Subscribed topic: `gcl.reco_requests`

### Required environment variables
Create a `.env` file in the project root with at least:

```env
KAFKA_BOOTSTRAP_SERVERS=kafka-host:kafka-port
```

### Local certificate requirements
The consumer expects these files relative to the project root:

* `certs/kafka-ca.pem`
* `certs/kafka-service.cert`
* `certs/kafka-service.key`

### Expected logs
When running, you should see:

* `Kafka consumer started...`
* `Received: ...` for incoming messages
* `Kafka consumer shut down gracefully.` when the app stops

### Troubleshooting
* Missing `KAFKA_BOOTSTRAP_SERVERS`: verify `.env` exists and is loaded.
* SSL/certificate errors: verify cert files exist and match the configured Kafka cluster.
* No messages consumed: confirm producers are publishing to `gcl.reco_requests` and offsets/group state are expected.
* Multiple consumers in development: FastAPI dev setups can spawn multiple processes; avoid running multiple app instances unless intended.
