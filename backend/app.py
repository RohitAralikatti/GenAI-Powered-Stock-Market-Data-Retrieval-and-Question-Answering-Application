# backend/app.py

from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .preprocess import preprocess_row, load_artifacts
from .predict import predict_proba_and_label
from backend.shap_explain_api import router as explain_router
from .shap_explain_api import shap_explain_single

app = FastAPI(title="FinTech GenAI Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SampleInput(BaseModel):
    features: Dict[str, float]


@app.get("/")
def root():
    return {"status": "API running", "message": "FinTech GenAI Backend Active"}


@app.post("/predict")
def predict(body: SampleInput):
    model, _, _, feature_names, _ = load_artifacts()

    for f in feature_names:
        if f not in body.features:
            body.features[f] = 0.0

    X_df = preprocess_row(body.features)
    preds, probs = predict_proba_and_label(model, X_df)

    return {
        "prediction": int(preds[0]),
        "probabilities": probs[0].tolist()
    }


app.include_router(explain_router)


from backend.langchain_engine.llama_client import generate_explanation

@app.post("/analyze")
def analyze(body: SampleInput):

    model, scaler, imputer, feature_names, explainer = load_artifacts()

    for name in feature_names:
        if name not in body.features:
            body.features[name] = 0.0

    X_df = preprocess_row(body.features)

    preds, probs = predict_proba_and_label(model, X_df)

    shap_res = shap_explain_single(
        explainer,
        model,
        X_df.values,
        feature_names
    )

    llm_explanation = generate_explanation(
        prediction=preds[0],
        probabilities=probs[0],
        top_features=shap_res["top_features"],
        shap_summary=shap_res["shap_summary"],
    )

    return {
        "prediction": int(preds[0]),
        "probabilities": probs[0].tolist(),
        "top_features": shap_res["top_features"],
        "shap_summary": shap_res["shap_summary"],
        "llm_explanation": llm_explanation,
    }

