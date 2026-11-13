#!/usr/bin/env bash
# Assemble the worker image from an existing PyTorch environment.
set -euo pipefail
if [[ $# != 1 ]]; then
  printf 'Usage: %s PYTHON_ENVIRONMENT_PREFIX\n' "$0" >&2
  exit 2
fi
PREFIX=$(realpath "$1")
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGING="$ROOT/.work/runtime-image"
mkdir -p "$STAGING/conda" "$STAGING/graft_worker"
"$PREFIX/bin/python" -c 'import torch; assert torch.__version__.split("+")[0] == "2.5.0"; print(torch.__version__, torch.version.cuda)'
# Copy environment files while keeping dependency caches out of the image.
tar -C "$PREFIX" --exclude='./pkgs' --exclude='./envs' --exclude='*/__pycache__' --exclude='*.pyc' -cf - . | tar -C "$STAGING/conda" -xf -
cp -a "$ROOT/python/graft_worker/." "$STAGING/graft_worker/"
printf '**/__pycache__\n**/*.pyc\n' > "$STAGING/.dockerignore"
cat > "$STAGING/Dockerfile" <<'DOCKERFILE'
FROM ubuntu:22.04@sha256:74b12b11ffa550fba1929ad4d8d16533c34f6612728469671afba1b03b66881c
COPY conda /opt/conda
COPY graft_worker /app/graft_worker
ENV PATH=/opt/conda/bin:$PATH PYTHONPATH=/app PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app
USER 10001:10001
DOCKERFILE
docker build -t graft-worker:0.1.0 "$STAGING"
docker image inspect graft-worker:0.1.0 --format '{{.Id}}'
