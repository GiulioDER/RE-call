FROM python:3.14-slim

WORKDIR /opt/recall
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-calc \
        libreoffice-impress \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*
COPY . /opt/recall
RUN pip install --no-cache-dir ".[mcp,fastembed,documents]"
