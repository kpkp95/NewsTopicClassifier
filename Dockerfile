FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-deploy.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.9.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ src/
COPY app.py .
COPY models/best_model/ models/best_model/

EXPOSE 8080

CMD ["python", "app.py"]
