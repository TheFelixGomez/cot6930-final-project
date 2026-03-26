from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


@patch("app.main.start_consumer")
@patch("app.main.stop_consumer")
def test_root_endpoint(mock_stop, mock_start):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World!"}
    
    mock_start.assert_called_once()
    mock_stop.assert_called_once()


@patch("app.main.start_consumer")
@patch("app.main.stop_consumer")
def test_ping_endpoint(mock_stop, mock_start):
    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}
    
    mock_start.assert_called_once()
    mock_stop.assert_called_once()