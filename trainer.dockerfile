# Base image
FROM python:3.11-slim


# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/exam_project/ src/exam_project/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/exam_project/train_model.py"]
