from langchain_community.llms import Ollama


def get_llm():
    """
    Returns a LangChain LLM wrapper around the local Llama model running in Ollama.
    """
    return Ollama(model="llama3.1")



def generate_explanation(
    prediction: int,
    probabilities: list,
    top_features: list,
    shap_summary: str,
):
    """
    Convert model + SHAP output into a GenAI-style explanation.
    """

    # --- probabilities ---
    prob_0, prob_1 = probabilities
    confidence = max(prob_0, prob_1)

    # --- direction ---
    direction = "outperformance" if prediction == 1 else "underperformance"

    # --- build feature lines (for prompt / logging) ---
    feature_lines = []
    for f in top_features[:5]:
        feature_lines.append(
            f"- {f['feature']} (impact: {f['shap_value']:.4f})"
        )

    # --- optional LLM prompt (future use) ---
    prompt = f"""
You are a financial analyst explaining a machine learning prediction.

Prediction: {direction}
Confidence: {confidence:.2%}

Top contributing features:
{chr(10).join(feature_lines)}

SHAP summary:
{shap_summary}

Write a concise, professional explanation suitable for a portfolio manager.
Avoid technical jargon.
"""

    # --- SAFE fallback explanation (NO INDEX ERRORS) ---
    top_feats = [f["feature"] for f in top_features[:3]]
    features_text = ", ".join(top_feats) if top_feats else "key macroeconomic factors"

    explanation = (
        f"The model predicts {direction} with approximately "
        f"{confidence:.1%} confidence. The prediction is primarily driven by "
        f"macroeconomic indicators such as {features_text}, "
        f"which suggest changing market conditions. "
        f"{shap_summary}"
    )

    return explanation
