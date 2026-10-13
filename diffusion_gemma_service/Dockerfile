FROM python:3.12-slim

WORKDIR /app

# Create non-root user for principle of least privilege
RUN useradd -m -u 10001 appuser

COPY telco_160_intents.json /app/telco_160_intents.json
COPY app.py /app/app.py

USER appuser

ENV PORT=8080
ENV BIND_HOST=0.0.0.0

EXPOSE 8080

CMD ["python3", "/app/app.py"]
