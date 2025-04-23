FROM python:3.12-slim

WORKDIR /app

# Cài đặt Rust và công cụ build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Cài rustup để lấy Rust compiler
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Tạo môi trường ảo Python
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Nâng cấp pip và cài maturin trong môi trường ảo
RUN pip install --upgrade pip
RUN pip install maturin

# Cài các thư viện Python trong requirements.txt (nếu có)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Build extension Rust bằng maturin (giả định có Cargo.toml)
RUN maturin develop

# Render yêu cầu cổng từ biến PORT
EXPOSE $PORT

# Khởi chạy bằng uvicorn trong môi trường ảo
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}