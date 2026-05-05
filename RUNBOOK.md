# Drift Triage Co-Pilot ML Runbook

## Local MLflow Workflow

Install dependencies:

```bash
python3 -m pip install -r requirements-ml.txt
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

## Notes

- Override `MLFLOW_TRACKING_URI` to target a different backend, such as `http://mlflow:5000`, when later Docker Compose wiring is added.
- Phase 2 only registers the trained candidate model and its metadata. Production promotion is intentionally deferred to a later gated phase.
