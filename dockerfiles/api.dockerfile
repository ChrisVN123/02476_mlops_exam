# Base image
FROM python:3.11-slim

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements_api.txt .
COPY pyproject.toml .
COPY src/ src/
COPY setup.py .
COPY README.md .
COPY requirements.txt .
COPY requirements_dev.txt .
COPY models/ models/
COPY data.dvc .

#RUN pip install -r requirements.txt --no-cache-dir
RUN dvc pull --no-run-cache
RUN pip install -r requirements_api.txt --no-cache-dir

#RUN pip install -e .
#RUN pip install . --no-deps --no-cache-dir
#ENTRYPOINT ["python", "-u", "src/exam_project/train.py"]

ENTRYPOINT ["uvicorn", "src.exam_project.predict:app", "--host", "0.0.0.0", "--port", "8000"]
