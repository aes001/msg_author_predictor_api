from app import app
from fastapi.testclient import TestClient
from .conf import client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
