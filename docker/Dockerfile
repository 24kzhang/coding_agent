FROM node:22-bookworm-slim AS node

FROM python:3.12-bookworm

ARG UV_VERSION=0.11.6

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && apt-get update \
    && apt-get install --yes --no-install-recommends build-essential ca-certificates curl git \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

ENV CI=1 \
    PYTHONUTF8=1 \
    HOME=/tmp/coding-agent-home \
    UV_CACHE_DIR=/tmp/coding-agent-cache/uv \
    NPM_CONFIG_CACHE=/tmp/coding-agent-cache/npm

WORKDIR /workspace

CMD ["/bin/sh", "-c", "mkdir -p \"$HOME\" \"$UV_CACHE_DIR\" \"$NPM_CONFIG_CACHE\" && exec sleep infinity"]
