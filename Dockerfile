FROM python:3.11-slim

LABEL maintainer="vijaymohan.akunuri@gmail.com"
LABEL description="Databricks QC MCP HTTP Server"

# Create non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY configs/ configs/
COPY mcp_server/ mcp_server/
COPY agents/ agents/
COPY llm/ llm/
COPY pipelines/ pipelines/
COPY schemas/ schemas/
COPY deploy/start_server.sh deploy/start_server.sh

RUN chmod +x deploy/start_server.sh

# Create log directory
RUN mkdir -p /var/log && chown appuser:appuser /var/log

# Switch to non-root user
USER appuser

EXPOSE 8000

ENV PROXY_MODE=true
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info
ENV WORKERS=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["./deploy/start_server.sh"]
