# Base image
FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
COPY pyproject.toml .
COPY src/ src/
COPY setup.py .
COPY README.md .
COPY requirements.txt .
COPY requirements_dev.txt .
COPY models/ models/
COPY data/ data/


RUN pip install --default-timeout=300 -r requirements_api.txt --no-cache-dir

ENTRYPOINT ["uvicorn", "src.exam_project.predict:app", "--host", "0.0.0.0", "--port", "8000"]
