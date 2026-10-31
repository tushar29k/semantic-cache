# slim image, no build tools needed — stdlib + fastapi only
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY semantic_cache/ ./semantic_cache/

# the service stores in CACHE_DB_PATH (default cache.db in /app);
# mount a volume there if the cache should survive a restart
EXPOSE 8000
CMD ["uvicorn", "semantic_cache.service:app", "--host", "0.0.0.0", "--port", "8000"]
