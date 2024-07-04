from fastapi import APIRouter, HTTPException
from app.authorPredictor.predictor import Predictor
import urllib.request
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()
predictorStore = []
STOP_WORDS = urllib.request.urlopen(
    "https://www.github.com/igorbrigadir/stopwords/blob/master/en/postgresql.txt").read().decode("utf-8").split("\n")

if os.getenv("MODEL_PATH") == None or os.getenv("DEFAULT_MODEL_NAME") == None:
    model_path = "testData/dummyModel"
else:
    model_path = os.getenv("MODEL_PATH") + os.getenv("DEFAULT_MODEL_NAME")

if os.getenv("MODEL_PATH") == None or os.getenv("SECOND_MODEL_NAME") == None:
    model_path_second = "testData/dummyModel"
else:
    model_path_second = os.getenv(
        "MODEL_PATH") + os.getenv("SECOND_MODEL_NAME")


def initializeDefaultPredictor():
    predictor = Predictor(model_path=model_path + "_model.keras", tokenizer_json_path=model_path + "_tokenizer.json",
                          stopwords_list=STOP_WORDS, usernames_json_list=model_path + "_users.json")
    return predictor


def initializeSecondPredictor():
    predictor = Predictor(model_path=model_path_second + "_model.keras", tokenizer_json_path=model_path_second + "_tokenizer.json",
                          stopwords_list=STOP_WORDS, usernames_json_list=model_path_second + "_users.json")
    return predictor


predictorStore.append(initializeDefaultPredictor())
predictorStore.append(initializeSecondPredictor())


@router.get("/predict")
def predict(text: str, id: int):
    if id == None or id < 0 or id >= len(predictorStore):
        raise HTTPException(status_code=404, detail="Invalid predictor id")
    return predictorStore[int(id)].predict(text)
