# Deploy EmbeddingGemma 2 to Azure Container Apps

This guide deploys the GHCR image as an OpenDots embedding API. Run the Bash commands in Azure Cloud Shell. **Before starting, verify the selected Azure subscription and understand that a free allowance is not a hard billing cap.**

## 0. Verify prerequisites

1. Confirm `ghcr.io/gokul7904231/embeddinggemma2` is **Public** in GitHub Packages. Azure can pull it without registry credentials only if it is public.
2. Sign in to [Azure Portal](https://portal.azure.com) and open Cloud Shell (Bash).
3. Check the subscription that will be billed:

```bash
az account show --query '{subscription:name,subscriptionId:id,tenantId:tenantId}' -o table
```

If this is not the subscription you intend to use, select the right one in the Azure portal or with `az account set --subscription "<subscription-id>"`.

## 1. Create a Consumption environment without Log Analytics ingestion

```bash
set -euo pipefail

LOCATION="centralindia"
RG="rg-opendots"
ENV_NAME="env-opendots"
APP_NAME="embeddinggemma2-api"
IMAGE="ghcr.io/gokul7904231/embeddinggemma2:latest"

az extension add --name containerapp --upgrade --yes
az provider register --namespace Microsoft.App --wait

az group create --name "$RG" --location "$LOCATION" --output table

if ! az containerapp env show --name "$ENV_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  az containerapp env create \
    --name "$ENV_NAME" \
    --resource-group "$RG" \
    --location "$LOCATION" \
    --logs-destination none
fi
```

The environment uses `--logs-destination none` to avoid creating a Log Analytics ingestion bill by default. This means centralized container logs are not configured; temporarily enable a logging destination only if you need it and have checked its price.

## 2. Generate a strong API key and create the app

Run only when the Container App does not already exist. The key is printed once so you can copy it into OpenDots' local `.env` file. Treat it as a password; don't paste it into GitHub, issues, or chat.

```bash
if az containerapp show --name "$APP_NAME" --resource-group "$RG" >/dev/null 2>&1; then
  echo "The app already exists. Do not recreate it or rotate its API key unintentionally."
else
  EMBED_API_KEY="$(openssl rand -hex 32)"

  az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --environment "$ENV_NAME" \
    --image "$IMAGE" \
    --target-port 8000 \
    --ingress external \
    --cpu 2.0 \
    --memory 4.0Gi \
    --min-replicas 0 \
    --max-replicas 1 \
    --secrets "embed-api-key=$EMBED_API_KEY" \
    --env-vars EMBED_API_KEY=secretref:embed-api-key MODEL_DTYPE=float32 MAX_BATCH_SIZE=16 \
    --query properties.configuration.ingress.fqdn \
    --output tsv

  printf '\nCopy this API key into OpenDots .env now:\n%s\n' "$EMBED_API_KEY"
  unset EMBED_API_KEY
fi
```

If the command reports the image is unauthorized or cannot be pulled, recheck the GHCR package visibility. Do not embed registry credentials in source code.

## 3. Get the hostname and test health

```bash
FQDN="$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

printf 'API host: https://%s\n' "$FQDN"
curl --fail --show-error "https://$FQDN/health"
```

Expected health response:

```json
{"status":"ok","model":"embeddinggemma-2"}
```

A healthy response proves the container started and loaded the model. Next test a real authenticated embedding request.

## 4. Smoke-test embeddings

Set `EMBED_API_KEY` in this shell to the key you generated (do not put it into command history). Then run:

```bash
read -rsp "Paste the API key: " EMBED_API_KEY; echo
curl --fail --show-error --silent \
  "https://$FQDN/v1/embeddings" \
  -H "Authorization: Bearer $EMBED_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":"OpenDots EmbeddingGemma provider smoke test","model":"embeddinggemma-2","task":"query","dimensions":768}' \
  | python -c 'import json,sys; d=json.load(sys.stdin); print("model:",d["model"]); print("dimensions:",len(d["data"][0]["embedding"])); print("normalized:",round(sum(x*x for x in d["data"][0]["embedding"]),4))'
unset EMBED_API_KEY
```

Expected: model `embeddinggemma-2`, dimensions `768`, and normalized vector squared-norm close to `1.0`.

## 5. Connect OpenDots

Add these entries to OpenDots' **local, untracked** `.env` file (substitute the hostname and key you saved):

```dotenv
AZURE_EMBED_BASE_URL=https://YOUR_CONTAINER_APP_FQDN/v1
AZURE_EMBED_API_KEY=YOUR_RANDOM_EMBED_API_KEY

OPENAI_EMBEDDING_BASE_URL=https://YOUR_CONTAINER_APP_FQDN/v1
OPENAI_EMBEDDING_API_KEY=YOUR_RANDOM_EMBED_API_KEY
OPENAI_EMBEDDING_MODEL=embeddinggemma-2
```

Restart OpenDots after saving the file. The OpenAI-compatible client should call `POST /v1/embeddings`. Check your actual OpenDots provider implementation before assuming it reads these variables; the source repository for the local `smart_router.mjs` was not available in this hosting repository.

Use 768 dimensions for both indexing and querying. If the existing OpenDots index was created with another model, do not mix the embeddings: rebuild or isolate the index.

## 6. Keep the free-tier risk bounded

- `--min-replicas 0`: permit scale-to-zero.
- `--max-replicas 1`: prevent scaling this app to multiple replicas.
- `--cpu 2.0 --memory 4.0Gi`: a 25-hour monthly ceiling at that size *if* all Azure Container Apps free grants remain unused.
- `--logs-destination none`: avoids default Log Analytics ingestion but reduces observability.
- Do not attach a custom virtual network, paid storage, or other billable resources without checking their prices.

The Consumption plan free grants are 180,000 vCPU-seconds, 360,000 GiB-seconds, and 2 million HTTP requests per subscription each calendar month. They're shared across eligible apps in the subscription. Usage beyond them can be billed; scale-to-zero only stops replica compute charges while no replicas are running. A Cost Management budget sends alerts and is not a hard spending stop.

## 7. If you update the image later

The repository workflow publishes both `:latest` and a commit-SHA tag. Prefer the immutable commit tag for a reproducible deployment. Then:

```bash
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --image "ghcr.io/gokul7904231/embeddinggemma2:COMMIT_SHA_FROM_SUCCESSFUL_BUILD" \
  --min-replicas 0 \
  --max-replicas 1
```

## Roll back / stop spending

To stop the service while retaining the resource definition, set minimum replicas to zero (already configured) and do not send requests. To remove the deployment and its environment when no longer needed:

```bash
az containerapp delete --name "$APP_NAME" --resource-group "$RG" --yes
az containerapp env delete --name "$ENV_NAME" --resource-group "$RG" --yes
az group delete --name "$RG" --yes --no-wait
```

The final command deletes every resource inside the resource group. Inspect the group first and do not use this cleanup sequence if it contains anything else.
