# Dev Caddie - Architecture Overview

## System Overview
Dev Caddie is a distributed hybrid system designed for low-latency voice briefings and high-signal content curation.

```
Browser UI
  ↕ Cloud Run (FastAPI + UI)
      ↕ (Internal service call)
        Sidecar Runtime (Pipecat + Daily + Gemini Live)
  ↕
Airflow (DAG)
  ↕
BigQuery (scored articles + daily briefing scripts)
```

## Core Responsibilities

- **Cloud Run**: UI + API gateway. Creates Daily rooms, starts/stops sidecar, serves briefing scripts and chat results.
- **Sidecar Runtime**: Pipecat + Daily WebRTC + Gemini Live audio. Handles real-time audio and barge‑in gating.
- **Airflow**: RSS ingestion, scoring, ranking, and daily briefing script generation.
- **BigQuery**: Persistent storage for scored articles and daily scripts.

## Data Pipeline (Airflow + BigQuery)

1. Fetch active feeds from metadata.
2. Pull RSS articles.
3. Deduplicate by normalized URL hash.
4. Score with Gemini + community signals.
5. Store into `articles_scored`.
6. Generate daily briefing script into `daily_briefings`.

## Scoring (Dual Scoring)

Final score blends AI relevance and community signals to avoid “obscure blog post” bias.

```
weights = get_adaptive_weights(content_age, community_signal_strength)
```

## Smart Feed Chat Assistant (StruQ)

Gemini extracts **structured intent** only (no SQL generation), which maps to parameterized queries. This prevents prompt injection while enabling natural language search.

## Morning Briefing (Deterministic + Q&A)

- **Deterministic script**: greeting + 5 summaries + closing.
- **Gemini Live** reads verbatim to avoid hallucinations.
- **Barge‑in gate** controls interruptions (OFF during narration, ON during Q&A).

## Observability

- **Briefing metrics**: TTFB, tokens, turns, interruptions.
- **Scoring metrics**: batch stats and coverage.
- **Repo observability**: cost + latency dashboards.

## Deployment (Public)

- **Cloud Run**: `deploy_dev_caddie.sh`
- **Sidecar**: internal deployment details are private, but Cloud Run communicates via a simple `/start` + `/stop` API.
