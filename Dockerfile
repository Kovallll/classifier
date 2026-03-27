FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Должен совпадать с ML_PORT / ML_SERVICE_URL в docker-compose (по умолчанию 50051)
EXPOSE 50051

CMD ["python", "server.py"]