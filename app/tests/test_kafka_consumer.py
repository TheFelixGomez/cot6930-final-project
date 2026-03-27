from unittest.mock import patch, MagicMock
from app.kafka import consumer


def test_start_stop_consumer():
    # Mock the thread so we don't actually spin up background tasks
    with patch("app.kafka.consumer.threading.Thread") as mock_thread:
        consumer.start_consumer()
        mock_thread.assert_called_once()
        assert consumer._run_flag is True
        
        consumer.stop_consumer()
        assert consumer._run_flag is False
        mock_thread.return_value.join.assert_called_once()


@patch("app.kafka.consumer.Consumer")
@patch("app.kafka.consumer.config")
@patch("app.kafka.consumer.os.path.exists")
def test_consume_loop(mock_exists, mock_config, mock_consumer_class):
    # Mock the environment to simulate local certs
    mock_exists.return_value = False
    mock_config.return_value = "localhost:9092"
    
    mock_consumer_instance = MagicMock()
    mock_consumer_class.return_value = mock_consumer_instance
    
    # Create a mock message
    mock_msg = MagicMock()
    mock_msg.error.return_value = False
    mock_msg.value.return_value = b"test message"
    
    # This function simulates polling a message AND turning off the loop
    # so we don't get stuck in an infinite while loop during tests
    def mock_poll(*args, **kwargs):
        consumer._run_flag = False
        return mock_msg
    
    mock_consumer_instance.poll.side_effect = mock_poll
    
    # Trigger the loop manually
    consumer._run_flag = True
    consumer.consume_loop()
    
    # Verify the Kafka functions were executed
    mock_consumer_class.assert_called_once()
    mock_consumer_instance.subscribe.assert_called_once_with(['gcl.reco_requests'])
    mock_consumer_instance.poll.assert_called_once()
    mock_consumer_instance.close.assert_called_once()


@patch("app.kafka.consumer.Consumer")
@patch("app.kafka.consumer.config")
@patch("app.kafka.consumer.os.path.exists")
def test_consume_loop_edge_cases(mock_exists, mock_config, mock_consumer_class):
    # 1. Hit the 'True' path for Render secrets
    mock_exists.return_value = True
    mock_config.return_value = "localhost:9092"
    
    mock_consumer_instance = MagicMock()
    mock_consumer_class.return_value = mock_consumer_instance
    
    # 2. Hit the msg is None and msg.error() paths
    mock_error_msg = MagicMock()
    mock_error_msg.error.return_value = True
    
    poll_responses = [None, mock_error_msg]
    
    def mock_poll(*args, **kwargs):
        if poll_responses:
            return poll_responses.pop(0)
        consumer._run_flag = False  # Shut down the loop safely
        return MagicMock(error=lambda: False, value=lambda: b"done")
    
    mock_consumer_instance.poll.side_effect = mock_poll
    
    consumer._run_flag = True
    consumer.consume_loop()
