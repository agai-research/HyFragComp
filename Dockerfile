# Reproducible environment for HyFragComp:  docker build -t hyfragcomp .  &&  docker run --rm -v "$PWD/results:/app/results" hyfragcomp ./reproduce.sh smoke
FROM python:3.12.3-slim
ENV PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["./reproduce.sh", "smoke"]
