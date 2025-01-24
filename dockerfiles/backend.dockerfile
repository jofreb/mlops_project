# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies required for Python packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
# WORKDIR /app

WORKDIR /src/nrms_ml_ops

# Set the PYTHONPATH to include the src directory
ENV PYTHONPATH=/src

# Copy the entire source directory to the container
COPY src src

# Install Python dependencies
COPY src/nrms_ml_ops/requirements_backend.txt requirements_backend.txt
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install --upgrade pip
RUN pip install python-multipart
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -r requirements_backend.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Expose the application port
EXPOSE $PORT

# Use JSON-style CMD for better signal handling
CMD ["uvicorn", "nrms_ml_ops.backend:app", "--host", "0.0.0.0", "--port", "8000"]
