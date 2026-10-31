FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY route_risk/ route_risk/
COPY train.py route.py analyze.py ./

ENTRYPOINT ["python"]
