from app import app
from fastapi.testclient import TestClient
import pytest
from app.authorPredictor.predictor import Predictor
import urllib.request
import json
import os


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def predictor():
    participant_count = 0
    participants = None
    with open("testData/dummyModel_users.json") as users_file:
        temp = json.load(users_file)
        participants = temp['participants']
        participant_count = len(temp['participants'])

    STOP_WORDS = urllib.request.urlopen(
        "https://github.com/igorbrigadir/stopwords/blob/master/en/postgresql.txt").read().decode("utf-8").split("\n")
    predictor = Predictor(model_path="testData/dummyModel_model.keras", tokenizer_json_path="testData/dummyModel_tokenizer.json",
                          stopwords_list=STOP_WORDS, usernames_json_list="testData/dummyModel_users.json")
    return predictor, participant_count, participants
