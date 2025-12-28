# backend/shap_explain_api.py

import numpy as np
import shap
from typing import Dict, List
from fastapi import APIRouter
from pydantic import BaseModel

from .preprocess import preprocess_row, load_artifacts
from .predict import predict_proba_and_label

router = APIRouter(tags=["Explain"])


class SampleInput(BaseModel):
    features: Dict[str, float]


def shap_explain_single(explainer, model, X: np.ndarray, feature_names: List[str], top_k: int = 5):
    if explainer is None:
        raise RuntimeError("SHAP explainer is not initialized")

    shap_vals = explainer.shap_values(X)
    shap_arr = np.array(shap_vals)

    if shap_arr.ndim == 3:
        shap_arr = shap_arr[1]

    if shap_arr.ndim == 2:
        shap_arr = shap_arr.reshape(1, -1)

    abs_vals = np.abs(shap_arr[0])
    top_idx = np.argsort(-abs_vals)[:top_k]

    top_features = [
        {
            "feature": feature_names[i],
            "shap_value": float(shap_arr[0][i]),
            "abs_shap": float(abs_vals[i])
        }
        for i in top_idx
    ]

    summary = (
        "Top increasing: "
        + ", ".join(f"{f['feature']} ({f['shap_value']:+.4f})" for f in top_features if f["shap_value"] > 0)
        + ". Top decreasing: "
        + ", ".join(f"{f['feature']} ({f['shap_value']:+.4f})" for f in top_features if f["shap_value"] < 0)
        + "."
    )

    #return top_features, summary
    return {
        "top_features": top_features,
        "shap_summary": summary,
    }



@router.post("/explain")
def explain(body: SampleInput):
    model, scaler, imputer, feature_names, explainer = load_artifacts()

    # fill missing features
    for f in feature_names:
        if f not in body.features:
            body.features[f] = 0.0

    X_df = preprocess_row(body.features)
    preds, probs = predict_proba_and_label(model, X_df)

    top_features, shap_summary = shap_explain_single(
        explainer,
        model,
        X_df.values,
        feature_names
    )

    return {
        "prediction": int(preds[0]),
        "probabilities": probs[0].tolist(),
        "top_features": top_features,        # ✅ REAL LIST
        "shap_summary": shap_summary,         # ✅ REAL STRING
        "llm_explanation": shap_summary       # ✅ TEMP: same text
    }
