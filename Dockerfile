FROM python:3.11-slim

ARG REQUIREMENTS_FILE=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt requirements-gpu-cu124.txt requirements-common.txt ./
RUN pip install --upgrade pip && pip install -r "${REQUIREMENTS_FILE}"

COPY main.py logging_utils.py ./
COPY controller ./controller
COPY worker ./worker

CMD ["python", "-u", "/app/worker/worker.py"]
