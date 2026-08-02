FROM python:3.12-slim 

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpoppler-cpp-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
    
# # Create virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

# Set working directory
WORKDIR /app
COPY . .

# Expose port
# EXPOSE 8001
ARG VLLM_PORT

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV VLLM_PORT=${VLLM_PORT}

# Run the FastAPI application
CMD python ./rest/llm_client_sessions.py