# Base image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .
COPY requirements_dev.txt .
COPY README.md .
COPY pyproject.toml .
COPY src/exam_project/ .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["uvicorn", "src/exam_project/api:app", "--host", "0.0.0.0", "--port", "8000"]
