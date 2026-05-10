FROM python:3.12-slim

WORKDIR /app

# This app serves static files with required COOP/COEP headers via server.py.
COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8082

EXPOSE 8082

CMD ["python", "server.py"]