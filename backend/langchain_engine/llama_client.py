from langchain_community.llms import Ollama


def get_llm():
    """
    Returns a LangChain LLM wrapper around the local Llama model running in Ollama.
    """
    return Ollama(model="llama3.1", timeout=30)


def generate_explanation(
    prediction: int,
    probabilities: list,
    top_features: list,
    shap_summary: str,
):
    """
    Convert model + SHAP output into a GenAI-style explanation.

    Tries the RAG-grounded Ollama pipeline first; falls back to a templated
    explanation if the LLM is unavailable (e.g. Ollama isn't running).
    """
    try:
        from .explain_chain import generate_rag_explanation

        return generate_rag_explanation(prediction, probabilities, top_features, shap_summary)
    except Exception:
        return _template_explanation(prediction, probabilities, top_features, shap_summary)


def _template_explanation(
    prediction: int,
    probabilities: list,
    top_features: list,
    shap_summary: str,
) -> str:
    prob_0, prob_1 = probabilities
    confidence = max(prob_0, prob_1)
    direction = "outperformance" if prediction == 1 else "underperformance"

    top_feats = [f["feature"] for f in top_features[:3]]
    features_text = ", ".join(top_feats) if top_feats else "key macroeconomic factors"

    return (
        f"The model predicts {direction} with approximately "
        f"{confidence:.1%} confidence. The prediction is primarily driven by "
        f"macroeconomic indicators such as {features_text}, "
        f"which suggest changing market conditions. "
        f"{shap_summary}"
    )
