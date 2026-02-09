#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="dev-caddie"
REGION="${REGION:-us-central1}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID not set. Export PROJECT_ID or set gcloud config."
  exit 1
fi
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com}"
DATASET_ID="${DATASET_ID:-content_intelligence}"
LLM_DATASET_ID="${LLM_DATASET_ID:-llm_observability}"
GCS_BUCKET_NAME="${GCS_BUCKET_NAME:-dev-caddie-media-${PROJECT_ID}}"
ADMIN_API_KEY_SECRET_NAME="${ADMIN_API_KEY_SECRET_NAME:-ADMIN_API_KEY}"
DAILY_BUDGET_USD="${DAILY_BUDGET_USD:-2.00}"
MAX_REQUESTS_PER_IP="${MAX_REQUESTS_PER_IP:-10}"
VACATION_MODE="${VACATION_MODE:-false}"
VACATION_END_DATE="${VACATION_END_DATE:-}"
CONTENT_API_BASE="${CONTENT_API_BASE:-}"
BOT_START_URL="${BOT_START_URL:-}"
BOT_START_ORIGIN="${BOT_START_ORIGIN:-}"
VPC_NETWORK="${VPC_NETWORK:-}"
VPC_SUBNET="${VPC_SUBNET:-}"
VPC_EGRESS="all-traffic"
CPU_ALWAYS_ALLOCATED="true"
REQUEST_TIMEOUT="600s"
CREATE_VPC_SUBNET="true"
CREATE_CLOUD_NAT="true"
CLOUD_ROUTER_NAME="cloud-run-router"
CLOUD_NAT_NAME="cloud-run-nat"

echo "=== Dev Caddie - Cloud Run Deployment ==="
echo "Mapping secrets:"
echo "  DAILY_API_KEY_SECRET_NAME=DAILY_API_KEY"
echo "  PROJECT_ID=${PROJECT_ID}"
echo "  DATASET_ID=${DATASET_ID}"
echo "  LLM_DATASET_ID=${LLM_DATASET_ID}"
echo "  GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
echo "  ADMIN_API_KEY_SECRET_NAME=${ADMIN_API_KEY_SECRET_NAME}"
echo "  DAILY_BUDGET_USD=${DAILY_BUDGET_USD}"
echo "  MAX_REQUESTS_PER_IP=${MAX_REQUESTS_PER_IP}"
echo "  VACATION_MODE=${VACATION_MODE}"
echo "  VACATION_END_DATE=${VACATION_END_DATE}"
echo "  CONTENT_API_BASE=${CONTENT_API_BASE}"
echo "  BOT_START_URL=${BOT_START_URL}"
echo "  BOT_START_ORIGIN=${BOT_START_ORIGIN}"
echo "  VPC_NETWORK=${VPC_NETWORK}"
echo "  VPC_SUBNET=${VPC_SUBNET}"
echo "  VPC_EGRESS=${VPC_EGRESS}"
echo "  CPU_ALWAYS_ALLOCATED=${CPU_ALWAYS_ALLOCATED}"
echo "  REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "  CREATE_VPC_SUBNET=${CREATE_VPC_SUBNET}"
echo "  CREATE_CLOUD_NAT=${CREATE_CLOUD_NAT}"
echo "  CLOUD_ROUTER_NAME=${CLOUD_ROUTER_NAME}"
echo "  CLOUD_NAT_NAME=${CLOUD_NAT_NAME}"
echo "  IMAGE_NAME=${IMAGE_NAME}:latest"
echo "  SERVICE_ACCOUNT=${SERVICE_ACCOUNT}"

echo "Building and pushing image with cache..."
gcloud builds submit --config cloudbuild.yaml --substitutions=_IMAGE=${IMAGE_NAME}

VPC_SUBNET_RANGE="${VPC_SUBNET_RANGE:-}"

if [[ "${CREATE_VPC_SUBNET}" == "true" ]]; then
  echo "Ensuring VPC subnet ${VPC_SUBNET} exists..."
  gcloud compute networks subnets describe "${VPC_SUBNET}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud compute networks subnets create "${VPC_SUBNET}" \
    --network "${VPC_NETWORK}" \
    --range "${VPC_SUBNET_RANGE}" \
    --region "${REGION}" \
    --enable-private-ip-google-access \
    --project "${PROJECT_ID}"
fi

if [[ "${CREATE_CLOUD_NAT}" == "true" ]]; then
  echo "Ensuring Cloud Router ${CLOUD_ROUTER_NAME} exists..."
  gcloud compute routers describe "${CLOUD_ROUTER_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud compute routers create "${CLOUD_ROUTER_NAME}" \
    --network "${VPC_NETWORK}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}"

  echo "Ensuring Cloud NAT ${CLOUD_NAT_NAME} exists..."
  gcloud compute routers nats describe "${CLOUD_NAT_NAME}" \
    --router "${CLOUD_ROUTER_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud compute routers nats create "${CLOUD_NAT_NAME}" \
    --router "${CLOUD_ROUTER_NAME}" \
    --region "${REGION}" \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges \
    --project "${PROJECT_ID}"
fi

gcloud run deploy "$SERVICE_NAME" \
  --image "${IMAGE_NAME}:latest" \
  --region "$REGION" \
  --allow-unauthenticated \
  --service-account "${SERVICE_ACCOUNT}" \
  --clear-secrets \
  --set-env-vars DAILY_API_KEY_SECRET_NAME=DAILY_API_KEY,ADMIN_API_KEY_SECRET_NAME=${ADMIN_API_KEY_SECRET_NAME},PROJECT_ID=${PROJECT_ID},DATASET_ID=${DATASET_ID},LLM_DATASET_ID=${LLM_DATASET_ID},GCS_BUCKET_NAME=${GCS_BUCKET_NAME},DAILY_BUDGET_USD=${DAILY_BUDGET_USD},MAX_REQUESTS_PER_IP=${MAX_REQUESTS_PER_IP},VACATION_MODE=${VACATION_MODE},VACATION_END_DATE=${VACATION_END_DATE},CONTENT_API_BASE=${CONTENT_API_BASE},BOT_START_URL=${BOT_START_URL},BOT_START_ORIGIN=${BOT_START_ORIGIN} \
  --network "${VPC_NETWORK}" \
  --subnet "${VPC_SUBNET}" \
  --vpc-egress "${VPC_EGRESS}" \
  --timeout "${REQUEST_TIMEOUT}" \
  $( [[ "${CPU_ALWAYS_ALLOCATED}" == "true" ]] && echo "--no-cpu-throttling" )
