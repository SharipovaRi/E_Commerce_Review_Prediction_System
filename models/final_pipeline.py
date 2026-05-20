import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

# Load trained sklearn pipeline
def load_model():
    return joblib.load(MODEL_PATH)

# Single sample dictionary
def predict(model, input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]


def predict_proba(model, input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict_proba(df)[0]