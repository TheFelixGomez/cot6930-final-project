# Real-Time Movie Recommendation Service

## Project Overview
This repository contains a real-time movie recommendation service built with FastAPI, Kafka, and Docker. It serves predictions and processes real-time user engagement streams.

## Repository Structure (subject to change)
* `/app`: FastAPI application (`app.py`, routes, schema validation).
* `/stream`: Kafka consumers and producers (`consumer.py`).
* `Dockerfile`: Container definition for the web service.
* `pyproject.toml`: Pinned Python dependencies.
* `.github/workflows`: CI/CD pipelines for testing and deployment.

## Development Setup (subject to change)
1. Clone the repository.
2. Install `uv` if not already installed: `pip install uv`
3. Create a virtual environment and install dependencies: `uv sync`
4. Copy `.env.example` to `.env` and fill in Kafka/API credentials.
5. Run the server locally: `uv run fastapi dev app/app.py`
