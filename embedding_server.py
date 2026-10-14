import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Union, List, Dict, Any
from sentence_transformers import SentenceTransformer

app = FastAPI(title="EmbeddingGemma-2 API")

EXPECTED_KEY = os.getenv("EMBED_API_KEY", "opendots-secret-key-123")

print("[*] Downloading / Loading EmbeddingGemma-2...")
# Loads google/embeddinggemma-2 and drops the audio encoder to save RAM (440M total size)
model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"audio_config": None}
)
print("[*] Model loaded successfully!")

TASK_MAP = {
    "query": "SearchQuery",
    "document": "Document",
    "code_query": "CodeRetrieval",
    "similarity": "SentenceSimilarity",
    "classification": "Classification",
}

class EmbeddingRequest(BaseModel):
    input: Union[str, List[Union[str, Dict[str, Any]]]]
    model: str = "embeddinggemma-2"
    task: str = "query"
    dimensions: int = 768

@app.get("/")
def health():
    return {"status": "online", "model": "google/embeddinggemma-2 (Audio omitted)"}

@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest, authorization: str = Header(default="")):
    if EXPECTED_KEY and authorization != f"Bearer {EXPECTED_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    inputs = [req.input] if not isinstance(req.input, list) else req.input
    is_media = any(isinstance(x, dict) for x in inputs)

    prompt_name = None if is_media else TASK_MAP.get(req.task)
    truncate_dim = req.dimensions if req.dimensions in (128, 256, 512, 768) else None

    vectors = model.encode(
        inputs,
        prompt_name=prompt_name,
        truncate_dim=truncate_dim,
        normalize_embeddings=True,
    )

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(vectors)
        ],
        "model": req.model,
    }
