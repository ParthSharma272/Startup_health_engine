# 🚀 Startup Health Engine

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![Apache Airflow](https://img.shields.io/badge/Orchestration-Airflow-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)  
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)  
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

An AI-powered platform that analyzes business documents (PDF, TXT, images) to extract KPIs and compute a standardized startup health score. The project uses a Streamlit UI as the frontend and an Airflow-orchestrated pipeline for processing and scoring.

## Table of contents

- [What's included](#whats-included)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Quickstart — local (recommended)](#quickstart)
- [Running Streamlit locally (dev)](#running-streamlit-locally-dev)
- [Environment variables & configuration](#environment-variables--configuration)
- [Deployment options](#deployment-options)
- [Architecture & how it works](#architecture--how-it-works)
- [Operational considerations](#operational-considerations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

## What's included

- `docker-compose.yaml` — main Compose stack for Airflow, Postgres, Redis, Streamlit, mlflow, workers.
- `Dockerfile` — used by Airflow-related services.
- `streamlit/app.py` — Streamlit UI (primary frontend today).
- `dags/` — Airflow DAGs for the pipeline.
- `processed_data/`, `uploads/`, `mlruns/`, `ml_models/` — data & model artifacts (volume-backed in Compose).

## Features (expanded)

This project implements a complete pipeline from document ingestion to an interpretable startup health score. Key capabilities:

- Document ingestion and OCR
	- Upload PDFs, images (PNG/JPG) and plain text via the Streamlit UI.
	- OCR using Tesseract and PDF parsing via `pdfminer.six`.

- Automated KPI extraction
	- Extracts financial and non-financial KPIs (revenue, growth, burn rate, runway, team size, traction signals, product metrics, etc.).
	- Uses rule-based extraction and optional LLM-assisted extraction when OpenAI is configured.

- Normalization and benchmarking
	- Normalizes raw KPI values to comparable scales using configurable benchmarks and percentile thresholds (see `config/` files).
	- Computes category-level scores and an aggregate startup health score (0–100).

- Confidence & prediction metadata
	- Generates a confidence score for each prediction and records which method produced the value (rule-based, ML, LLM).

- Actionable insights & recommendations
	- Combines rule-based alerts and AI-assisted recommendations to surface strengths, weaknesses, and prioritized next steps.

- Orchestration & observability
	- Airflow DAGs manage end-to-end processing with retries, logging and task-level visibility.
	- MLflow tracks model experiments, artifacts, and model versions.

- Export & integrations
	- Download full analysis as JSON.
	- Extensible outputs for BI tools or downstream systems (CSV/JSON/MLflow artifacts).

## Tech stack (details & core versions)

- Frontend: Streamlit (app at `streamlit/app.py`)
- Orchestration: Apache Airflow (tested with `apache-airflow[cncf.kubernetes]==2.11.0`)
- Containers: Docker & Docker Compose
- Database: PostgreSQL (metadata)
- Broker: Redis (Celery broker)
- ML tracking: MLflow
- Important Python libraries (as listed in `requirements.txt`):
	- pandas, numpy, scikit-learn
	- plotly (visualization)
	- transformers==4.42.3, torch==2.3.1 (optional LLM/embedding workloads)
	- pdfminer.six==20221105, Pillow==10.3.0, pytesseract==0.3.10 (document processing)
	- openai, sentence-transformers (LLM and embedding integrations)
- Optional runtime: Gunicorn for WSGI services

Notes: exact pinned versions are maintained in `requirements.txt`; in production consider using explicit pins and a lockfile.

## Quickstart

Run the full stack locally with Docker Compose (recommended for development/testing).

### Prerequisites

- Docker & Docker Compose installed
- Optional: Python 3.10+ for local Streamlit development

### Start the stack (from repository root)

```bash
# 1) Start core infra (Postgres + Redis)
docker compose up -d postgres redis

# 2) Initialize Airflow (this runs DB init and creates the admin user)
docker compose up --build airflow-init

# 3) Start the remaining services
docker compose up -d --build

# 4) Confirm services are running
docker compose ps
```

Web UIs:

- Airflow: http://localhost:8080 (default admin/admin created by `airflow-init`)
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5001

## Running Streamlit locally (dev)

For faster iteration you can run the Streamlit app locally without Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit/app.py
```

## Environment variables & configuration

Recommended environment variables (use a `.env` loaded by Compose or set in your shell):

- `AIRFLOW_API_BASE_URL` — Airflow REST API base (default: `http://airflow-webserver:8080/api/v1`)
- `AIRFLOW_UI_BASE_URL` — Airflow UI URL for links in the app
- `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` — credentials used by the Streamlit app to trigger DAGs
- `OPENAI_API_KEY` — (optional) set to enable LLM suggestions and embedding calls
- `MLFLOW_TRACKING_URI` — optional override if you host MLflow elsewhere

Configuration files:

- `config/` contains JSON files for KPI weights, benchmarks and percentile thresholds. Tweak these to adapt scoring to different industries.

## Deployment options

Choose based on scale and operational preferences:

- Single VM with Docker Compose + Traefik (recommended for small teams)
	- Simple to operate, supports persistent volumes and Traefik-managed TLS.
	- Good for proof-of-concept and internal deployments.

- Managed / Kubernetes (for production scale)
	- Use the official Airflow Helm chart, managed Postgres (RDS/Cloud SQL), object storage (S3/Spaces) and an Ingress with cert-manager.
	- Scales better for many concurrent DAGs and workers.

- PaaS options for Streamlit only
	- Streamlit Cloud, Render or Vercel (if you migrate to Next.js). If you host Streamlit separately, ensure the backend APIs remain reachable and secure.

## Architecture & how it works

High-level flow:

1. User uploads a document in the Streamlit UI. The file is saved to `uploads/` (shared volume).
2. The UI triggers an Airflow DAG run with the file name in `conf`.
3. Airflow tasks extract text (OCR/parsing), run KPI extraction (rules + models), normalize scores, and persist results to `processed_data/` as JSON.
4. Streamlit polls `processed_data/` and displays results (scores, KPI tables, Plotly charts). The user can download the full JSON.

Data & observability:

- Airflow logs are written to `logs/` and visible via the Airflow UI.
- MLflow stores experiments in `mlruns/` (local by default).

## Operational considerations

- Secrets & credentials: do NOT commit credentials. Use environment variables, Docker secrets, or a secrets manager.
- Backups: schedule periodic Postgres dumps and back up `mlruns/` if experiments matter.
- Storage: for production, replace local volume artifact storage with S3 (or S3-compatible) to simplify scaling and backups.
- Monitoring: add Prometheus/Grafana or a cloud provider monitoring for CPU/Memory and Airflow task failures.
- Scaling: increase Airflow workers and move to Kubernetes if you need horizontal scaling and task isolation.

## Troubleshooting

- Port conflicts: run `lsof -nP -iTCP:8501 -sTCP:LISTEN` to locate processes blocking Streamlit.
- Airflow init errors: check `docker compose logs airflow-init` and ensure Postgres/Redis are healthy.
- Missing Python deps: `pip install -r requirements.txt` in a venv for local runs.


## Roadmap

- Improve Streamlit UI (UX polish, metric cards, theme toggle)
- Add authenticated, server-side APIs to reduce direct exposure of Airflow
- Add CI/CD: build/publish images and automations to deploy to a Linux host or registry

## License

This project is released under the MIT License. See `LICENSE` for details.

