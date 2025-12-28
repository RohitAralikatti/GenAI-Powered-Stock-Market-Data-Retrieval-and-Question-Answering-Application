from langchain_core.prompts import PromptTemplate
from .llama_client import get_llm
from .vectorstore import get_vectorstore

"""
This module generates natural-language explanations for model predictions
using your local Llama 3.1 model via Ollama.
"""

# ---------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------

EXPLAIN_PROMPT = """
You are a financial analysis assistant.

A machine learning model predicted: {prediction_value}

Explain in simple, clear terms why this prediction *might* have occurred,
based on the following features:

{feature_info}

Your explanation must be:
- easy to understand for a finance student,
- concise,
- based on logical reasoning,
- NOT overly confident (no guaranteed statements).

Begin your explanation now:
"""

prompt = PromptTemplate(
    input_variables=["prediction_value", "feature_info"],
    template=EXPLAIN_PROMPT
)

# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def generate_explanation(prediction_value: float, feature_info: str):
    """
    Sends the prompt to Llama 3.1 and returns the generated explanation.
    """
    llm = get_llm()

    # Format prompt
    final_prompt = prompt.format(
        prediction_value=prediction_value,
        feature_info=feature_info
    )

    # Generate explanation
    response = llm.invoke(final_prompt)

    return response
