# ml-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Установка Python пакетов
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Генерация gRPC кода
RUN python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    categorizer.proto

RUN mkdir -p /app/models

EXPOSE 50051

CMD ["python", "server.py"]