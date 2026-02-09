# Pipecat Briefing Bot

This folder contains a Pipecat-based replacement for the custom Live API WebSocket flow.
It keeps the existing code intact for reminiscence, while providing a new Daily/WebRTC
pipeline.

## Prereqs
- Daily API key (stored in Secret Manager)
- GCP credentials with Vertex AI access (ADC or service account)

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r pipecat/requirements.txt
```

## Run (local)
```bash
python pipecat/briefing_bot.py
```

Pipecat will prompt for a Daily room URL/token or create one depending on runner args.
Refer to Pipecat Daily transport docs for provisioning details.

## Subprocess Bot (bot.py)
If you use the Cloud Run `/api/briefing/start` endpoint, it spawns `pipecat/bot.py`
as a subprocess with the room URL + token.

The bot includes a function-calling tool `get_briefing_data` that fetches LLM-related
articles from your existing `/api/feeds/daily_reads` endpoint.

Environment variables:
- `GOOGLE_CLOUD_PROJECT` (GCP project ID)
- `CONTENT_API_BASE` (default: http://localhost:8080)
