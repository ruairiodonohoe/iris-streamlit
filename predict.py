import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[0]
MODEL_DIR = BASE_DIR / "resources" / "models"
MODEL_NAME = "rf_model.sav"
MODEL_FILEPATH = MODEL_DIR / MODEL_NAME

def predict(data):
    clf = joblib.load(MODEL_FILEPATH)
    return clf.predict(data)