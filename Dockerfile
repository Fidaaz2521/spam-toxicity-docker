FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')"

CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "7860"]
ENV HF_SPACE_HEALTH_CHECK_TIMEOUT=120
