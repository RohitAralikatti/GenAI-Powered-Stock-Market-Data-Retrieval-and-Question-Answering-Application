# backend/preprocess.py

import json
import joblib
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
import shap

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "catboost_model.cbm"))

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")

    with open(MODEL_DIR / "feature_list.json", "r") as f:
        feature_names = json.load(f)

    explainer = shap.TreeExplainer(model)

    return model, scaler, imputer, feature_names, explainer


def preprocess_row(features: dict) -> pd.DataFrame:
    _, scaler, imputer, feature_names, _ = load_artifacts()

    # Auto-fill missing features
    for f in feature_names:
        if f not in features:
            features[f] = 0.0

    df = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)

    df = pd.DataFrame(imputer.transform(df), columns=feature_names)
    df = pd.DataFrame(scaler.transform(df), columns=feature_names)

    return df
