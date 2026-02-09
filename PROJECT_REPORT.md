# Dev Caddie - Project Report (Judges Edition)

> An AI-powered tech briefing system that curates, scores, and narrates the most relevant engineering reads each day.

## About the Project

### Inspiration
I built Dev Caddie because I kept missing high‑signal engineering posts in the noise of daily feeds. I wanted a short, trustworthy briefing that feels like a personal caddie: it picks the best reads, tells me why they matter, and lets me ask questions live.

The inspiration also came from the “latency gap” in current AI assistants. Models are smart, but the transition from reasoning to real‑time conversation often feels disjointed. I wanted a system where the caddie could process complex inputs and respond with the immediacy of a human partner.

### What it does
- Turns 500+ daily engineering articles into a short voice briefing
- Blends Gemini relevance with community signals for high‑signal picks
- Supports live Q&A via barge‑in gating
- **Smart Feeds**: ranked feed dashboard and curated reading lists
- **Feed Assistant**: natural‑language search over the full scored corpus (BigQuery)

### How we built it
A distributed hybrid architecture separates UI orchestration from real‑time execution:
- **Cloud Run:** UI + API surface and session orchestration
- **GCE VM + Pipecat:** real‑time audio streaming with Gemini Live
- **Airflow + BigQuery:** ingestion, scoring, ranking, and briefing script generation

### Challenges we ran into
- **Barge‑in gating:** letting users interrupt during Q&A but not during scripted narration required an in‑house gate mechanism (`allow_interruptions`) and real‑time UI sync.
- **Handshake reliability:** keeping Daily + Gemini Live stable across refreshes and session restarts required strict teardown and room cleanup.
- **Security and identity isolation:** metadata‑based auth without hardcoded secrets.
- **Latency optimization:** tracked TTFA with

\[
L_{total} = T_{network} + T_{inference} + T_{processing} < 800ms
\]

Optimizing the sidecar network path and metadata fetch reduced \(T_{processing}\). A latency graph will be added to the Observability tab.

### Accomplishments that we're proud of
- Deterministic briefing + live Q&A without context resets
- Real-time barge-in gating synchronized to the UI
- Dual-scoring that resists “obscure blog post” bias
- **Cost control:** Budget guard caps spend at $2/day with graceful degradation. Rate limiting enforces fairness so one user can’t exhaust the free Gemini quota.

### What we learned
- **Sidecars are powerful:** offloading real‑time media handling keeps the main app lean.
- **Orchestration matters:** prompt design is only part of the problem; data flow is the rest.
- **Modular openness:** expose interfaces while protecting the proprietary scoring pipeline.

### What's next for Dev Caddie
- Expand voice personalization and topic controls
- Explore graph‑based context (knowledge graph / topological memory) for richer AI grounding
- **Live Article Lookup**: Enable users to ask follow-up questions like "tell me more about article 2" or "what's the URL for the Kubernetes story" via Gemini Live function calling, with real-time access to article metadata from BigQuery
- **User Favorites**: Allow users to mark articles as favorites, feeding explicit relevance signals back into the knowledge graph to personalize future scoring and briefing selection

---

## Executive Summary

Dev Caddie turns 500+ daily technical articles into a short, voice-first briefing. It combines **AI relevance scoring** with **community signals** (HN/Lobsters) to avoid surface-level picks, then narrates the top stories through a Gemini Live + Daily voice stack with barge‑in controls.

**What makes it different:**
- **Dual Scoring**: AI relevance + community signals (prevents “obscure blog post” bias)
- **Narrated Briefing**: Gemini Live reads a generated script, not ad‑hoc hallucinations
- **Barge‑in Control**: Real-time gating so users can interrupt during Q&A but not during scripted delivery
- **Operationally Lean**: ~$20–35/month, single agent, no heavy infra

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI on Cloud Run |
| AI (Scoring, Chat, Briefing Script) | Gemini 3 Flash Preview |
| Live Audio | Gemini Live 2.5 Flash (native audio) |
| Data Storage | BigQuery |
| Rate Limiting | Firestore |
| Observability | Cloud Monitoring (latency graph pending) |

## Demo Flow (2 minutes)

1. Open Dev Caddie → Click **Start**
2. Briefing begins (Gemini Live voice)
3. Say “next” to advance to the next article
4. Ask a question mid‑briefing → Gemini answers briefly
5. End session

---

## Architecture Overview

Dev Caddie has three user-facing experiences, each backed by the same scored‑articles corpus:
1. **Daily Voice Briefing** (deterministic script + barge‑in gating)
2. **Smart Feed Chat Assistant** (NL → BigQuery results)
3. **Smart Feeds Dashboard** (ranked feeds and analytics)

**How Smart Feed numbers are derived**
- **AI score** = `ai_relevance_score` (Gemini batch scoring in Airflow)
- **Community score** = `community_score` (HN/Lobsters points/comments → weighted total)
- **Final score** = `final_score` (AI + community with confidence weights + viral override)
- **Water Cooler** shows raw HN/Lobsters points/comments instead of AI/Final

```
RSS Feeds (OPML)
  → Airflow DAG
    → Scoring (Gemini + HN/Lobsters)
      → BigQuery (articles_scored)
        ├─ Daily Briefing Script → Cloud Run API → Sidecar VM (Pipecat + Daily + Gemini Live) → Browser UI
        ├─ Smart Feed Chat (StruQ) → Cloud Run API → BigQuery results → Browser UI
        └─ Smart Feeds Dashboard → Cloud Run API → Browser UI
```

**Key services:**
- **Airflow DAG**: ingestion, dedup, scoring, ranking
- **Cloud Run**: API + UI
- **VM Sidecar**: Pipecat + Daily WebRTC + Gemini Live

---

# Key Innovations

## 1) Dual-Scoring System (AI + Community)
AI alone can’t distinguish between a niche tutorial and a battle‑tested engineering post. Dev Caddie fixes that by blending:
- **AI relevance (Gemini)**: semantic fit with engineering interests
- **Community signals**: HN/Lobsters scores and decay trends

Result: high‑signal picks without manual curation.

**Model:** `gemini-2.5-flash` (global)

**Viral override:** community spikes can lift a post, but an AI relevance floor prevents off‑topic viral content from taking over.

**Metric lineage (Smart Feeds UI):** AI/Community/Final scores are pulled from `articles_scored` in BigQuery and rendered per card; HN/Lobsters points are shown for the Water Cooler feed.

This solves the **“obscure blog post”** problem—where AI alone can’t distinguish between a random tutorial and a battle‑tested Netflix engineering post.

## 2) Voice Briefing Pipeline (Deterministic → Q&A)
The briefing is **not** an open-ended chat. It’s a deterministic script built from the top 5 articles:
- **Greeting**
- **5 summaries** (one paragraph per URL)
- **Closing**

The hardest part was **switching between deterministic narration and nondeterministic Q&A** without breaking Gemini Live context. The system keeps the script flow rigid during narration, then opens the gate for live user questions once an article finishes.

**Models:**  
- Script generation: `gemini-2.5-flash` (global)  
- Live audio: `gemini-live-2.5-flash-native-audio` (us-central1)

Gemini Live reads the script verbatim to avoid hallucinations and keep timing predictable.

## 3) Barge‑In Messaging (Gate + UI Sync)
Interruptions are **disabled during scripted delivery** and enabled during Q&A. This prevents Gemini Live context resets.

- Gate: `allow_interruptions` (true/false)
- Push to UI via Pipecat `TransportMessageUrgentFrame` → Daily app-message:
  - `{ "type": "gate-status", "allow_interruptions": true|false, "reason": "..." }`
- UI displays dedicated badge ("Barge-in: On/Off") so users know when they can interject

## 4) Smart Feed Chat Assistant (BigQuery‑Backed)
Beyond the daily briefing, Dev Caddie ships a **Feed Assistant** that answers natural‑language questions against the full scored‑articles database in BigQuery.

**How it works:**
- **NL → Structured Intent (StruQ):** Gemini extracts a strict intent schema (topics, time range, min score, limit).
- **Safe SQL Templates:** No raw SQL generation. The system maps intent to parameterized BigQuery queries.
- **Results:** Returns scored articles with titles, summaries, links, and scoring breakdowns.

This provides a fast, transparent way to explore the corpus without exposing the scoring logic itself.

**Model:** `gemini-2.5-flash` (global)

**StruQ pattern (safe NL→SQL):** the assistant never generates SQL from user input. It extracts structured intent and maps to parameterized queries.

---

# Technical Appendix (Concise)

## A. Data Pipeline (Airflow)

**DAG Schedule:** Runs daily at `13:00 UTC` (`0 13 * * *`) to refresh feeds, score articles, and generate the morning briefing.

**OPML Feeds → BigQuery**
- OPML file defines all feeds
- Airflow reads active feeds from BigQuery

**Feed Health Tracking**
- Auto‑disable after 3 consecutive failures
- Error state stored in BigQuery

**Deduplication**
- SHA‑256 hash of normalized URLs
- Prevents duplicate scoring

**HN RSS Score Extraction**
- HN scores parsed from hnrss.org feed descriptions
- Reduces external API calls

## B. Scoring System

**Dual‑Score Formula**
- AI relevance score (Gemini)
- Community score (HN/Lobsters)
- Final score = weighted blend + decay schedule

**Model:** `gemini-2.5-flash` (global)

**Structured output:** Gemini returns type‑safe JSON validated by Pydantic before scoring and storage.

**Dynamic weights** based on content freshness and community validation:  
`weights = get_adaptive_weights(content_age, community_signal_strength)`

**Decay‑Based Rescoring**
- 16h → 24h → 7d → stop
- Prevents endless rescoring costs

## C. Briefing Generation

**Top‑5 Selection**
- After scoring, top 5 are selected by final score

**Summarization**
- Gemini creates one paragraph per URL
- Paragraphs are used verbatim in briefing script

**Model:** `gemini-2.5-flash` (global)

**Output**
- Stored in BigQuery and served by Cloud Run

## D. Voice Runtime (Sidecar VM)

**Why a VM Sidecar?**
- Stable long‑running WebRTC sessions
- Low latency for Gemini Live audio

**Stack**
- Pipecat + Daily WebRTC + Gemini Live
- Systemd service for reliability

**Model:** `gemini-live-2.5-flash-native-audio` (us-central1)

## E. Security & Safety

**Rate Limiting**
- IP‑based throttling to prevent abuse
- Daily budget limits for Gemini usage

**Prompt‑Injection Defense (5 layers)**

| Layer | Implementation |
|------|----------------|
| 1. Input Validation | Regex patterns block common attack vectors |
| 2. XML Wrapping | User input wrapped in `<user_input>` tags |
| 3. Structured Output | Strict JSON schema validated by Pydantic |
| 4. StruQ Pattern | Intent extraction only; no SQL generation |
| 5. Output Validation | Schema enforcement + prompt leakage checks |

## F. Operations

**Vacation Mode**
- Ultra‑strict limits when not actively monitoring

**Cost Profile (Monthly)**
- Gemini: ~$3
- Cloud Run: ~$0
- BigQuery/Firestore: ~$0
- VM: ~$15–20

## G. Video Lecture Notes

Video lecture notes are a supporting feature. The intent is to convert free YouTube/IV‑League course content into structured study notes:

- Uploads snapshots to GCS alongside generated notes
- Stores structured notes in BigQuery (`lecture_notes`)
- UI renders notes in the Lectures tab

Snapshot extraction is part of the lecture processing pipeline.

**Model:** `gemini-2.5-flash` (global)

---

## Repository Notes

- **Public view** can show UI + Cloud Run + architecture docs
- **Private components**: Airflow DAGs, scoring logic, sidecar runtime

---

*Last updated: 2026-02-07*
