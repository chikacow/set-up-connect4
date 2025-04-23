FROM python:3.12-slim

WORKDIR /app

# Cài đặt Rust và các công cụ build cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Rust thông qua rustup
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Cài đặt maturin để build Rust extension
RUN pip install --upgrade pip
RUN pip install maturin

# Cấu hình môi trường Python
ENV PYTHONUNBUFFERED=1

# Copy và cài đặt requirements nếu có
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy mã nguồn
COPY . .

# Build Rust extension (giả sử bạn có file Cargo.toml ở root dự án)
RUN maturin develop

# Sử dụng biến môi trường PORT (do Render cung cấp)
EXPOSE $PORT

# Khởi chạy ứng dụng với uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]