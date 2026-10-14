import io
import os
import torch
from fastapi import FastAPI, Header, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Union, List, Dict, Any
from PIL import Image
from sentence_transformers import SentenceTransformer

app = FastAPI(title="EmbeddingGemma-2 OpenAI + Multimodal API")

EXPECTED_KEY = os.getenv("EMBED_API_KEY", "opendots-secret-key-123")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model_440m")

print(f"[*] Loading pre-stripped 440M model from {MODEL_PATH}...")
model = SentenceTransformer(
    MODEL_PATH,
    model_kwargs={"torch_dtype": torch.bfloat16},
)
print("[*] Model loaded and ready!")

# Matches official EmbeddingGemma 2 Prompt Names from the Model Card
TASK_MAP = {
    "query": "SearchQuery",
    "document": "Document",
    "code_query": "CodeRetrieval",
    "qa": "QuestionAnswering",
    "fact_check": "FactChecking",
    "classification": "Classification",
    "clustering": "Clustering",
    "similarity": "SentenceSimilarity",
}

class EmbeddingRequest(BaseModel):
    input: Union[str, List[Union[str, Dict[str, Any]]]]
    model: str = "embeddinggemma-2"
    task: str = "query"
    dimensions: int = 768  # Supports MRL truncation: 768, 512, 256, 128

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model": "google/embeddinggemma-2 (440M Text+Code+Image)"}

@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest, authorization: str = Header(default="")):
    if EXPECTED_KEY and authorization != f"Bearer {EXPECTED_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    inputs = [req.input] if not isinstance(req.input, list) else req.input
    is_media = any(isinstance(x, dict) for x in inputs)

    prompt_name = None if is_media else TASK_MAP.get(req.task, "SearchQuery")
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
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

@app.post("/v1/embed/image")
async def embed_image(
    file: UploadFile = File(...),
    dimensions: int = Form(768),
    authorization: str = Header(default=""),
):
    if EXPECTED_KEY and authorization != f"Bearer {EXPECTED_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    truncate_dim = dimensions if dimensions in (128, 256, 512, 768) else None

    vec = model.encode(img, truncate_dim=truncate_dim, normalize_embeddings=True)
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": vec.tolist()}],
        "model": "embeddinggemma-2",
    }
