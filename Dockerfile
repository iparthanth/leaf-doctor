FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY knowledge knowledge
COPY model model
COPY web web
# the disk is wiped on every restart/redeploy: cases are a demo, not a record
ENV MODEL_DIR=/srv/model WEB_DIR=/srv/web DATA_DIR=/tmp/leafdoctor PORT=10000
EXPOSE 10000
# one worker: the rate limiter and the SQLite lock live in-process (and the free plan has 512 MB)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
