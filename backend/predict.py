# backend/predict.py

import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path

from .preprocess import preprocess_row, load_artifacts

MODEL_DIR = Path(__file__).resolve().parent / "model"


def get_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "catboost_model.cbm"))
    return model


def predict_proba_and_label(model, X_df: pd.DataFrame):
    probs = model.predict_proba(X_df)
    preds = probs.argmax(axis=1)
    return preds, probs


def predict_returns(df: pd.DataFrame):
    model, scaler, imputer, feature_names, _ = load_artifacts()

    X_df = df.copy()

    for f in feature_names:
        if f not in X_df.columns:
            X_df[f] = 0.0

    X_df = X_df[feature_names]

    X_df[:] = imputer.transform(X_df)
    X_df[:] = scaler.transform(X_df)

    preds = model.predict(X_df)

    return pd.DataFrame({"prediction": preds})
