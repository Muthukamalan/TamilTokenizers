FROM ghcr.io/astral-sh/uv:trixie-slim

RUN mkdir -p /home/tokeniz

WORKDIR /home/tokeniz

COPY pyproject.toml requirements.txt ./

# Install dependencies
RUN uv sync --all-groups

COPY minbpe ./minbpe/
COPY server.py ./

COPY models ./models/
EXPOSE 8000
CMD ["uv", "run", "python","-m" "serve"]
