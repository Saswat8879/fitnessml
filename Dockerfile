FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    wget \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN mkdir -p /app/models

ENV MODEL_PATH=/app/models/plate_ranker_lgb.pkl
EXPOSE 8000

CMD ["uvicorn", "model_server_sklearn:app", "--host", "0.0.0.0", "--port", "8000"]
