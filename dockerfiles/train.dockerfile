# Base image
FROM python:3.11.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# set the working directory in the container
# WORKDIR /app

# copy the dependencies file to the working directory
COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
# COPY data/processed data/processed
COPY models models

# install dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# set the command to run the application
ENTRYPOINT ["python", "-u", "src/nrms_ml_ops/train_cloud.py"]
