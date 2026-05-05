FROM python:3.11-slim

WORKDIR /app

COPY services/model_service/requirements.txt services/model_service/requirements.txt
RUN pip install --no-cache-dir -r services/model_service/requirements.txt

COPY services services
COPY artifacts artifacts

EXPOSE 8000

CMD ["uvicorn", "services.model_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

