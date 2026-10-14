# EmbeddingGemma 2 API

A small FastAPI service that exposes Google's `google/embeddinggemma-2` through an OpenAI-compatible `POST /v1/embeddings` endpoint. The container includes the text/vision model configuration used by this project; the audio encoder is omitted to reduce deployment size.

## Endpoints

- `GET /health` — unauthenticated health check.
- `POST /v1/embeddings` — text or supported multimodal inputs. Requires `Authorization: Bearer <EMBED_API_KEY>`.
- `POST /v1/embed/image` — image upload endpoint. Requires the same bearer key.

Example request:

```json
{
  "input": "OpenDots retrieval test",
  "model": "embeddinggemma-2",
  "task": "query",
  "dimensions": 768
}
```

Supported tasks: `query`, `document`, `code_query`, `qa`, `fact_check`, `classification`, `clustering`, and `similarity`. Supported output dimensions: 768, 512, 256, and 128. Embeddings are normalized.

## Security and runtime configuration

- `EMBED_API_KEY` is **required**. The service refuses to start without it; there is no default credential in source code.
- Store the key in your hosting platform's secret store. Never commit a real API key into Git or a public `.env` file.
- `MODEL_DTYPE` defaults to `float32`, a conservative choice for CPU hosting. EmbeddingGemma 2 supports `float32` and `bfloat16`; do not use `float16`.
- `MAX_BATCH_SIZE` defaults to 64.
- `MAX_IMAGE_BYTES` defaults to 10 MiB.

## Container image

```text
ghcr.io/gokul7904231/embeddinggemma2:latest
```

Use the immutable Git SHA tag for repeatable deployments. GHCR visibility must be Public if the platform pulls this image without registry credentials.

## Deploy to Azure Container Apps

Follow [AZURE_DEPLOY.md](AZURE_DEPLOY.md), or run `bash deploy-azure.sh` from Azure Cloud Shell for a guarded deployment and an authenticated 768-dimensional smoke test. The script asks you to confirm the subscription and explicitly approve resource creation, disables Log Analytics ingestion by default, and saves OpenDots settings with restrictive file permissions in your Cloud Shell home directory. Run it only after the latest GHCR workflow succeeds and the image package is Public. Azure's free allowances are not a hard spending cap.

## OpenDots configuration

After deployment, set these in OpenDots' local `.env` file and replace the hostname and key with the values from your deployment:

```dotenv
OPENAI_EMBEDDING_BASE_URL=https://YOUR_CONTAINER_APP_FQDN/v1
OPENAI_EMBEDDING_API_KEY=YOUR_RANDOM_EMBED_API_KEY
OPENAI_EMBEDDING_MODEL=embeddinggemma-2
```

The base URL ends at `/v1`; the OpenAI-compatible client should append `/embeddings`. Keep the same embedding model and dimensions for queries and indexed documents. If switching to a different model, do not compare its vectors with existing EmbeddingGemma vectors unless the index is rebuilt.

## Local container smoke test

Set `EMBED_API_KEY` in the shell, then start the image while passing the key as an environment variable. Test `GET /health` and call `POST /v1/embeddings` with a Bearer token. Do not expose this endpoint publicly without a strong key and request limits.

## License

Check Google's model license and usage terms before distributing or operating the model in a hosted service.
