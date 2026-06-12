from langchain_core.prompts import PromptTemplate

from .feature_descriptions import describe_feature
from .llama_client import get_llm
from .vectorstore import get_vectorstore, retrieve_context

"""
Generates natural-language explanations for model predictions using a
local Llama 3.1 model via Ollama, grounded with context retrieved from a
ChromaDB vector store of financial concept documents (RAG).
"""

EXPLAIN_PROMPT = """
You are a financial analysis assistant explaining a machine learning
prediction to a portfolio manager.

Prediction: {direction} (confidence: {confidence})

Top contributing factors:
{feature_lines}

SHAP summary: {shap_summary}

Background on these kinds of factors:
{retrieved_context}

Write a concise, professional explanation of why the model likely made
this prediction. Use plain language, avoid jargon, and do not make
guaranteed statements about future performance.
"""

prompt = PromptTemplate(
    input_variables=["direction", "confidence", "feature_lines", "shap_summary", "retrieved_context"],
    template=EXPLAIN_PROMPT,
)


def generate_rag_explanation(prediction: int, probabilities, top_features, shap_summary: str) -> str:
    """
    Builds a RAG-grounded prompt from the prediction, top SHAP features,
    and relevant financial-concept documents, then sends it to Llama 3.1
    via Ollama.
    """
    prob_0, prob_1 = probabilities
    confidence = f"{max(prob_0, prob_1):.1%}"
    direction = "outperformance" if prediction == 1 else "underperformance"

    feature_lines = "\n".join(
        f"- {describe_feature(f['feature'])} "
        f"({'pushes toward outperformance' if f['shap_value'] > 0 else 'pushes toward underperformance'}, "
        f"impact {f['shap_value']:+.4f})"
        for f in top_features[:5]
    )

    query_text = "; ".join(describe_feature(f["feature"]) for f in top_features[:5])
    collection = get_vectorstore()
    context_docs = retrieve_context(collection, query_text, k=3)
    retrieved_context = "\n\n".join(context_docs) if context_docs else "No additional context retrieved."

    final_prompt = prompt.format(
        direction=direction,
        confidence=confidence,
        feature_lines=feature_lines,
        shap_summary=shap_summary,
        retrieved_context=retrieved_context,
    )

    llm = get_llm()
    response = llm.invoke(final_prompt)

    return response.strip()
