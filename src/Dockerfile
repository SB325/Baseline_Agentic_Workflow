#FROM python:3.12-slim 
# FROM vllm/vllm-openai:latest
FROM vllm/vllm-openai:qwen38

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpoppler-cpp-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
    
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# Ensure uv is available in the PATH
# ENV PATH="/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

# Install dependencies using uv with Docker build caching
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install --system -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose port
# EXPOSE 8001
ARG VLLM_PORT

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV VLLM_PORT=${VLLM_PORT}

# Run the FastAPI application
ENTRYPOINT ["python3"]
CMD ["/app/rest/llm_client_sessions.py"]