from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.kafka.consumer import start_consumer, stop_consumer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Boot up: Start the Kafka background thread
    start_consumer()
    yield
    # Shut down: Signal the thread to stop and clean up
    stop_consumer()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/ping")
async def ping():
    return {"message": "pong"}
