# backend/langchain_engine/knowledge_base.py
#
# Financial concept documents used to seed the RAG vector store. Each
# document gives background context on a group of related features, so
# the LLM can ground its explanation in real financial reasoning rather
# than just restating SHAP numbers.

CONCEPT_DOCS = [
    {
        "id": "momentum",
        "text": (
            "Momentum factors (1-month, 6-month, 12-month, 36-month returns, "
            "change in momentum, and industry momentum) capture the tendency "
            "of stocks that have recently performed well to keep performing "
            "well in the near term, and for long-term winners to eventually "
            "reverse. Strong recent momentum is often read as a sign that "
            "positive sentiment or information is still being absorbed by "
            "the market."
        ),
    },
    {
        "id": "value",
        "text": (
            "Value factors (book-to-market, earnings-to-price, cash-flow-to-"
            "price, sales-to-price, dividend yield) compare a company's "
            "accounting fundamentals to its market price. High value scores "
            "suggest a stock is cheap relative to its fundamentals, which "
            "historically has been associated with higher average future "
            "returns but can also signal distress."
        ),
    },
    {
        "id": "profitability_quality",
        "text": (
            "Profitability and quality factors (gross profitability, "
            "operating profitability, return on equity/assets/invested "
            "capital, Piotroski and Mohanram scores, earnings growth "
            "streaks) measure how efficiently a company turns assets and "
            "equity into profit. Higher quality and profitability are "
            "generally associated with more durable future returns."
        ),
    },
    {
        "id": "investment_growth",
        "text": (
            "Investment and growth factors (asset growth, capital "
            "expenditure growth, inventory and sales growth, share "
            "issuance) describe how quickly a firm is expanding. Rapid "
            "asset growth or heavy share issuance has historically been "
            "associated with weaker subsequent returns, sometimes "
            "interpreted as a sign of overinvestment."
        ),
    },
    {
        "id": "risk_volatility",
        "text": (
            "Risk and volatility factors (market beta, beta squared, "
            "idiosyncratic volatility, return volatility, maximum daily "
            "return) describe how much a stock's price swings, both in "
            "line with the broader market and on its own. Higher volatility "
            "and higher beta generally mean a riskier, more "
            "market-sensitive stock."
        ),
    },
    {
        "id": "liquidity",
        "text": (
            "Liquidity factors (dollar trading volume, share turnover, "
            "bid-ask spread, the Amihud illiquidity measure, zero-trading "
            "days) capture how easily a stock can be bought or sold without "
            "moving its price. Illiquid stocks often carry a return premium "
            "to compensate investors for the extra trading cost and risk."
        ),
    },
    {
        "id": "leverage_financial_health",
        "text": (
            "Leverage and financial-health factors (leverage ratio, current "
            "and quick ratios, cash holdings, cash flow to debt, secured "
            "debt) describe a company's debt burden and ability to meet "
            "short-term obligations. Higher leverage increases financial "
            "risk, while strong liquidity and low debt suggest resilience."
        ),
    },
    {
        "id": "macro_environment",
        "text": (
            "Macroeconomic factors (market dividend yield, market "
            "earnings-to-price and book-to-market ratios, net equity "
            "issuance, the Treasury bill rate, the term spread, the default "
            "spread, and overall market volatility) describe the broader "
            "economic and interest-rate environment. A wider default "
            "spread or term spread typically signals tighter credit "
            "conditions and higher perceived economic risk, which tends to "
            "weigh on expected stock returns across the board."
        ),
    },
    {
        "id": "shap_interpretation",
        "text": (
            "SHAP values explain a model's prediction by attributing it to "
            "individual input features. A positive SHAP value means that "
            "feature pushed the prediction toward outperformance, while a "
            "negative SHAP value means it pushed the prediction toward "
            "underperformance. The magnitude of the SHAP value reflects how "
            "much influence that feature had on this specific prediction, "
            "not whether the feature is 'good' or 'bad' in general."
        ),
    },
]
