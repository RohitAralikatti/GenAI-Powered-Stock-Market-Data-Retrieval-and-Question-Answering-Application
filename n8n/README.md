# n8n Wrapper Workflow

This workflow wraps the FastAPI `/analyze` endpoint so it can be triggered
via webhook and (optionally) post results to Slack.

## Workflow

```
Webhook (POST /webhook/analyze)
  -> Call /analyze API   (HTTP Request -> http://localhost:8000/analyze)
  -> Format Result        (builds a human-readable summary string)
  -> Respond to Webhook    (returns prediction + SHAP summary + LLM explanation as JSON)
  -> Notify Slack (optional, disabled by default)
```

## Setup

1. Make sure the FastAPI backend is running on `http://localhost:8000`
   (`uvicorn backend.app:app --port 8000`).
2. In n8n: **Workflows -> Import from File** -> select `fintech-analyzer-workflow.json`.
3. If n8n is running in Docker, replace `localhost` in the
   "Call /analyze API" node with `host.docker.internal` so it can reach
   the host machine's port 8000.
4. Activate the workflow, then send a test request:

```bash
curl -X POST http://localhost:5678/webhook/analyze \
  -H "Content-Type: application/json" \
  -d '{"features": {"mvel1": 0.5, "beta": 1.2, "betasq": 1.44, "chmom": 0.1,
       "dolvol": 0.3, "idiovol": 0.2, "indmom": 0.05, "mom1m": 0.02,
       "mom6m": 0.08, "mom12m": 0.15}}'
```

5. (Optional) To enable Slack notifications: open "Notify Slack (optional)",
   add your Slack credentials and pick a channel, then re-enable the node.

## What this demonstrates

- Wrapping an existing AI/ML API (CatBoost + SHAP + RAG + Ollama) behind a
  low-code automation trigger.
- Webhook-driven automation suitable for chat ops / scheduled batch runs
  (pair with an n8n Cron node feeding a list of tickers into this workflow).
- Extending the result with downstream notifications (Slack/email) without
  touching the Python backend.
