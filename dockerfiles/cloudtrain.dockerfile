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
COPY configs/ configs/


#RUN pip install -r requirements.txt --no-cache-dir
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install your project as a package in editable mode
RUN pip install -e .

# Specify the entry point for running the training script
ENTRYPOINT ["python", "-m", "exam_project.train_model"]
