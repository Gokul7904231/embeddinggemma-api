import io
import os
import secrets
from typing import Any, Dict, List, Union

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="EmbeddingGemma 2 OpenAI-Compatible Embeddings API")

# Never ship a usable default credential in source control or the image.
EXPECTED_KEY = os.getenv("EMBED_API_KEY", "").strip()
if not EXPECTED_KEY:
    raise RuntimeError("EMBED_API_KEY is required. Configure it as a platform secret.")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model_440m")
MODEL_DTYPE_NAME = os.getenv("MODEL_DTYPE", "float32").lower()
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
if MODEL_DTYPE_NAME not in DTYPES:
    raise RuntimeError("MODEL_DTYPE must be 'float32' or 'bfloat16'.")

print(f"[*] Loading EmbeddingGemma 2 from {MODEL_PATH} with {MODEL_DTYPE_NAME}...")
model = SentenceTransformer(
    MODEL_PATH,
    model_kwargs={"torch_dtype": DTYPES[MODEL_DTYPE_NAME]},
)
print("[*] EmbeddingGemma 2 loaded and ready.")

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
SUPPORTED_DIMENSIONS = (128, 256, 512, 768)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[Union[str, Dict[str, Any]]]]
    model: str = "embeddinggemma-2"
    task: str = "query"
    dimensions: int = 768


def require_api_key(authorization: str) -> None:
    expected = f"Bearer {EXPECTED_KEY}"
    if not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_dimensions(dimensions: int) -> Union[int, None]:
    if dimensions not in SUPPORTED_DIMENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"dimensions must be one of {SUPPORTED_DIMENSIONS}",
        )
    return dimensions if dimensions != 768 else None


@app.get("/")
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": "embeddinggemma-2"}


@app.post("/v1/embeddings")
def create_embeddings(
    req: EmbeddingRequest,
    authorization: str = Header(default=""),
) -> Dict[str, Any]:
    require_api_key(authorization)

    inputs = [req.input] if not isinstance(req.input, list) else req.input
    if not inputs:
        raise HTTPException(status_code=422, detail="input must not be empty")
    if len(inputs) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"batch exceeds the configured limit of {MAX_BATCH_SIZE} inputs",
        )
    if any(isinstance(item, str) and not item.strip() for item in inputs):
        raise HTTPException(status_code=422, detail="input strings must not be empty")
    if req.task not in TASK_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"task must be one of {tuple(TASK_MAP.keys())}",
        )

    contains_multimodal_input = any(isinstance(item, dict) for item in inputs)
    prompt_name = None if contains_multimodal_input else TASK_MAP[req.task]
    truncate_dim = validate_dimensions(req.dimensions)

    vectors = model.encode(
        inputs,
        prompt_name=prompt_name,
        truncate_dim=truncate_dim,
        normalize_embeddings=True,
    )

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vector.tolist()}
            for i, vector in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/embed/image")
async def embed_image(
    file: UploadFile = File(...),
    dimensions: int = Form(768),
    authorization: str = Header(default=""),
) -> Dict[str, Any]:
    require_api_key(authorization)
    truncate_dim = validate_dimensions(dimensions)

    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds the {MAX_IMAGE_BYTES}-byte upload limit",
        )

    try:
        with Image.open(io.BytesIO(raw)) as source_image:
            image_rgb = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    vector = model.encode(
        image_rgb,
        truncate_dim=truncate_dim,
        normalize_embeddings=True,
    )
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": vector.tolist()}],
        "model": "embeddinggemma-2",
    }
