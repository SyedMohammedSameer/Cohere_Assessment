# Backend image for the Cohere Chat API.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the application and its dependencies.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

# Run as an unprivileged user that owns the working directory (the SQLite file
# is written here unless DATABASE_URL points elsewhere).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# COHERE_API_KEY must be supplied at runtime, for example:
#   docker run -e COHERE_API_KEY=... -p 8000:8000 cohere-chat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
