# FROM python:3.12-slim

# WORKDIR /app

# # Install system dependencies including Rust toolchain
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#     && export PATH="/root/.cargo/bin:$PATH" \
#     && rm -rf /var/lib/apt/lists/*

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app \
#     PATH="/root/.cargo/bin:${PATH}" \
#     VIRTUAL_ENV=/app/venv

# # Create and activate virtual environment
# RUN python -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     pip install maturin

# # Copy application code
# COPY . .

# # # Build Rust extension (corrected path to Rust project)
# # RUN if [ -f "conv/Cargo.toml" ]; then \
# #         cd conv && \
# #         echo "Building with opening book at: $(pwd)/opening_database.bin" && \
# #         [ -f "opening_database.bin" ] || echo "Warning: opening_database.bin not found!" && \
# #         maturin develop --release; \
# #     fi
# # RUN ls -la /app/conv/ && \
# #     [ -f "/app/conv/opening_database.bin" ] && \
# #     echo "Opening book found!" || echo "Opening book missing!"
# # Render yêu cầu cổng từ biến PORT
# EXPOSE $PORT

# # Khởi chạy bằng uvicorn trong môi trường ảo
# CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}


FROM python:3.12-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho Python (nếu có)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/venv

# Tạo và kích hoạt virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn ứng dụng
COPY . .

# Mở cổng từ biến môi trường PORT
EXPOSE $PORT

# Khởi chạy ứng dụng bằng uvicorn
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}