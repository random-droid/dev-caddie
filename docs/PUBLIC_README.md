# Public Repo Notes

This public repository intentionally omits private runtime and infrastructure code.

## Private Components (Not Included)

- **Sidecar runtime (`vm/`)**: systemd service, deployment scripts, and VM entrypoint that run Pipecat + Daily + Gemini Live.
- **Airflow DAGs (`airflow/`)** and scoring pipelines.
- **Data/infra (`bigquery/`, `terraform/`, `data/`)**.

## Sidecar API Contract (Private)

The Cloud Run service communicates with a private sidecar over an internal VPC IP.

Endpoints:
- `POST /start` → starts a briefing bot for a given Daily room.
- `POST /stop` → stops a briefing bot for a given room.
- `GET /health` → liveness probe.

Request payload (start):
```
{ "room_url": "https://...daily.co/ROOM", "token": "DAILY_TOKEN", "script": "..." }
```

The sidecar publishes audio via Daily WebRTC and pushes UI state over Daily app-messages.
