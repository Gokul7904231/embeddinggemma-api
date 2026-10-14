FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model_440m \
    MODEL_DTYPE=float32 \
    MAX_BATCH_SIZE=64 \
    MAX_IMAGE_BYTES=10485760

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image and omit the unused audio encoder. Float32 is
# the conservative default for CPU inference; never use float16 for this model.
RUN python -c "\
import torch; \
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer( \
    'google/embeddinggemma-2', \
    model_kwargs={'torch_dtype': torch.float32}, \
    config_kwargs={'audio_config': None} \
); \
model.save('/app/model_440m')" && rm -rf /root/.cache/huggingface

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
