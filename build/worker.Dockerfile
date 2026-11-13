FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
RUN python -m pip install --no-cache-dir numpy==2.1.3
WORKDIR /app
COPY python/graft_worker /app/graft_worker
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
USER 10001:10001
