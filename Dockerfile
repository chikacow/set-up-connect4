FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:/root/.cargo/bin:$PATH"

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install maturin

# Copy the full project code
COPY . .

# Build the Rust extension with maturin (if Cargo.toml exists)
RUN if [ -f "conv/Cargo.toml" ]; then \
        cd conv && \
        echo "Building Rust extension..." && \
        if [ -f "opening_database.bin" ]; then \
            echo "Found opening_database.bin"; \
        else \
            echo "⚠️  Warning: opening_database.bin not found!"; \
        fi && \
        maturin develop --release; \
    fi

# Optional: Check file exists
RUN ls -la /app/conv/ && \
    [ -f "/app/conv/opening_database.bin" ] && \
    echo "opening_database.bin loaded." || \
    echo  "opening_database.bin missing."

# Expose the port passed via environment
EXPOSE $PORT

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
