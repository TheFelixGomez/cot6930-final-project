import threading
from confluent_kafka import Consumer
from decouple import config

_run_flag = False
_consumer_thread = None

# NOTE: this will run successfully but due to FastAPI's default behavior of running multiple instances,
# we may want to consider using a more robust solution for Kafka consumption. (This is just for our project scope)
def consume_loop():
    global _run_flag
    
    conf = {
        'bootstrap.servers': config('KAFKA_BOOTSTRAP_SERVERS'),
        'group.id': 'gcl_consumer_group',
        'auto.offset.reset': 'earliest',
        'security.protocol': 'SSL',
        'ssl.ca.location': 'certs/kafka-ca.pem',
        'ssl.certificate.location': 'certs/kafka-service.cert',
        'ssl.key.location': 'certs/kafka-service.key',
    }
    
    consumer = Consumer(conf)
    consumer.subscribe(['gcl.reco_requests'])
    
    print("Kafka consumer started...")
    
    while _run_flag:
        # Poll briefly so the thread can exit cleanly on shutdown
        msg = consumer.poll(1.0)
        
        if msg is None:
            continue
        if msg.error():
            print(f"Kafka error: {msg.error()}")
            continue
        
        # Process the message here
        print(f"Received: {msg.value().decode('utf-8')}")
    
    consumer.close()
    print("Kafka consumer shut down gracefully.")


def start_consumer():
    global _run_flag, _consumer_thread
    _run_flag = True
    _consumer_thread = threading.Thread(target=consume_loop, daemon=True)
    _consumer_thread.start()


def stop_consumer():
    global _run_flag, _consumer_thread
    _run_flag = False
    if _consumer_thread:
        _consumer_thread.join()
