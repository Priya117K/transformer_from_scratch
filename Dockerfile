FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MODEL_PATH=/app/transformer_en_it.pt \
    TOKENIZER_EN_PATH=/app/data/tokenizer_en.json \
    TOKENIZER_IT_PATH=/app/data/tokenizer_it.json

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py train.py ./
COPY models ./models
COPY data ./data

EXPOSE 7860  

CMD ["python", "inference.py"]
