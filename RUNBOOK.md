# Drift Triage Co-Pilot ML Runbook

## Local MLflow Workflow

Install dependencies:

```bash
python3 -m pip install -r requirements-ml.txt
python3 -m pip install -r services/model_service/requirements.txt
```

Train the offline model and generate artifacts:

```bash
python3 -m ml.train
```

Register the existing artifacts in local MLflow:

```bash
python3 -m ml.register_mlflow
```

Open the local MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then visit:

```text
http://localhost:5000
```

Run the model service locally from the repo root:

```bash
uvicorn services.model_service.app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Model health check:

```bash
curl http://localhost:8000/health/model
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "job": "technician",
    "marital": "married",
    "education": "professional.course",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
  }'
```

## Notes

- Override `MLFLOW_TRACKING_URI` to target a different backend, such as `http://mlflow:5000`, when later Docker Compose wiring is added.
- Phase 2 only registers the trained candidate model and its metadata. Production promotion is intentionally deferred to a later gated phase.
- The Phase 3 Dockerfile is only a skeleton. Local live verification should run from the repo root where `./mlruns` exists. Docker runtime with `file:./mlruns` will require mounting or copying that store; later Docker Compose can use `MLFLOW_TRACKING_URI=http://mlflow:5000`.
