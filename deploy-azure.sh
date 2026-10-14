#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

LOCATION="centralindia"
RESOURCE_GROUP="rg-opendots"
ENVIRONMENT="env-opendots"
APP_NAME="embeddinggemma2-api"
IMAGE="ghcr.io/gokul7904231/embeddinggemma2:latest"
CONFIG_FILE="$HOME/opendots-embedding.env"

command -v az >/dev/null || { echo "Azure CLI (az) is required." >&2; exit 1; }
command -v openssl >/dev/null || { echo "openssl is required to generate a strong API key." >&2; exit 1; }

echo "Current Azure subscription:"
az account show --query '{subscription:name,id:id,tenant:tenantId}' --output table
SUBSCRIPTION_ID="$(az account show --query id --output tsv)"
read -r -p "Type this subscription ID to confirm resource creation ($SUBSCRIPTION_ID): " CONFIRMED_ID
[[ "$CONFIRMED_ID" == "$SUBSCRIPTION_ID" ]] || { echo "Subscription confirmation did not match; stopping."; exit 1; }

echo
echo "This creates a Central India Container Apps environment and a public HTTPS endpoint."
echo "The app scales to zero, max 1 replica; centralized logging is disabled."
echo "Free grants are not a hard billing cap. Other subscription usage can consume the shared allowance."
read -r -p "Type CREATE to proceed: " CONFIRM
[[ "$CONFIRM" == "CREATE" ]] || { echo "Cancelled; no resources created."; exit 0; }

if az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "The app already exists. Stopping to avoid rotating its API key or changing its configuration."
  echo "Inspect it with: az containerapp show -n $APP_NAME -g $RESOURCE_GROUP"
  exit 2
fi

az extension add --name containerapp --upgrade --yes
az provider register --namespace Microsoft.App --wait
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --tags app=opendots component=embeddinggemma2 --output none

if ! az containerapp env show --name "$ENVIRONMENT" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    --name "$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --logs-destination none \
    --output none
fi

EMBED_API_KEY="$(openssl rand -hex 32)"
FQDN=""

# Save endpoint settings with restrictive file permissions. Keep this file in
# Cloud Shell's home directory, not in the cloned repository.
FQDN_CONFIG="$HOME/opendots-embedding.env"
cat > "$FQDN_CONFIG" <<EOF
AZURE_EMBED_BASE_URL=https://PLACEHOLDER/v1
AZURE_EMBED_API_KEY=$EMBED_API_KEY
OPENAI_EMBEDDING_BASE_URL=https://PLACEHOLDER/v1
OPENAI_EMBEDDING_API_KEY=$EMBED_API_KEY
OPENAI_EMBEDDING_MODEL=embeddinggemma-2
EOF
chmod 600 "$FQDN_CONFIG"

az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENVIRONMENT" \
  --image "$IMAGE" \
  --target-port 8000 \
  --ingress external \
  --cpu 2.0 \
  --memory 4.0Gi \
  --min-replicas 0 \
  --max-replicas 1 \
  --secrets "embed-api-key=$EMBED_API_KEY" \
  --env-vars EMBED_API_KEY=secretref:embed-api-key MODEL_DTYPE=float32 MAX_BATCH_SIZE=16 \
  --output none

FQDN="$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

cat > "$FQDN_CONFIG" <<EOF
AZURE_EMBED_BASE_URL=https://$FQDN/v1
AZURE_EMBED_API_KEY=$EMBED_API_KEY
OPENAI_EMBEDDING_BASE_URL=https://$FQDN/v1
OPENAI_EMBEDDING_API_KEY=$EMBED_API_KEY
OPENAI_EMBEDDING_MODEL=embeddinggemma-2
EOF
chmod 600 "$FQDN_CONFIG"

echo "Waiting for the container to start and load the model (up to 15 minutes)..."
HEALTH_OK=0
for attempt in $(seq 1 90); do
  if curl --connect-timeout 5 --max-time 15 --fail --silent "https://$FQDN/health" >/dev/null; then
    HEALTH_OK=1
    break
  fi
  sleep 10
done

if [[ "$HEALTH_OK" -ne 1 ]]; then
  echo "The container did not become healthy in time. The deployment and secret file have been retained for debugging." >&2
  echo "Check: az containerapp revision list -n $APP_NAME -g $RESOURCE_GROUP -o table" >&2
  echo "Your secure config file is: $FQDN_CONFIG" >&2
  exit 3
fi

echo "Health check passed. Testing a real embedding request..."
read -r -d '' TEST_BODY <<'JSON' || true
{"input":"OpenDots EmbeddingGemma API smoke test","model":"embeddinggemma-2","task":"query","dimensions":768}
JSON
RESPONSE="$(curl --connect-timeout 5 --max-time 180 --fail --silent --show-error \
  "https://$FQDN/v1/embeddings" \
  -H "Authorization: Bearer $EMBED_API_KEY" \
  -H "Content-Type: application/json" \
  --data "$TEST_BODY")"

printf '%s' "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); v=d["data"][0]["embedding"]; norm=sum(x*x for x in v); print("model:",d["model"],"| dimensions:",len(v),"| squared_norm:",round(norm,4)); assert len(v)==768, "Expected a 768-dimension vector"; assert abs(norm-1.0)<0.02, "Vector does not look normalized"'

unset EMBED_API_KEY
echo
echo "Deployment and embedding smoke test succeeded."
echo "Secure OpenDots config saved at: $FQDN_CONFIG (permissions 600)."
echo "Copy its KEY=VALUE lines to OpenDots' local .env. Never commit that file."
echo "Base URL: https://$FQDN/v1"
echo
echo "Next: verify that OpenDots' embedding client actually reads these settings, then restart it."
