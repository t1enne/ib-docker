FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

RUN apt-get update && apt-get install -y \
	gcc \
	g++ \
	pkg-config \
	libffi-dev \
	python3-dev \ 
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

ENV HDF5_DIR=/usr

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv
# RUN uv sync --system
RUN uv pip install --only-binary=:all: zstandard

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

COPY . .

CMD ["python", "main.py"]
