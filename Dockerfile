FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py server.py rag_indexer.py dev_trace.py start.sh ./
COPY chroma_db/ ./chroma_db/

RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
