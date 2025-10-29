# 🚀 Startup Health Engine

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![Apache Airflow](https://img.shields.io/badge/Orchestration-Airflow-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)  
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)  
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

An AI-powered platform that analyzes business documents (PDF, TXT, images) to extract KPIs and compute a standardized startup health score. The project uses a Streamlit UI as the frontend and an Airflow-orchestrated pipeline for processing and scoring.

## Table of contents
- [What's included](#whats-included)
- [Quickstart — local (recommended)](#quickstart)
- [Running Streamlit locally (dev)](#running-streamlit-locally-dev)
- [Deployment options](#deployment-options)
- [Architecture & design notes](#architecture--design-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## What's included

- `docker-compose.yaml` — main Compose stack for Airflow, Postgres, Redis, Streamlit, mlflow, workers.
- `Dockerfile` — used by Airflow-related services.
- `streamlit/app.py` — Streamlit UI (primary frontend today).
- `dags/` — Airflow DAGs for the pipeline.
- `processed_data/`, `uploads/`, `mlruns/`, `ml_models/` — data & model artifacts (volume-backed in Compose).

## Quickstart

These instructions launch the full stack locally using Docker Compose (recommended for development/testing).

### Prerequisites
- Docker and Docker Compose (or Docker Compose plugin) installed
- (Optional) `python3`, a virtualenv if you want to run Streamlit outside Docker

### Start the stack (from repository root)

```bash
# 1) Start dependent services (postgres + redis) first
docker compose up -d postgres redis

# 2) Initialize Airflow (run until it completes; shows logs)
docker compose up --build airflow-init

# 3) Start the full stack (webserver, scheduler, worker, streamlit, mlflow)
docker compose up -d --build

# 4) Verify
docker compose ps
```

Open the UIs in your browser:
- Airflow:  http://localhost:8080  (default admin/admin created by init)
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5001

Notes
- If port 8501 (Streamlit) is already in use locally, stop the local process or change the `ports` mapping in `docker-compose.yaml` (e.g., `8502:8501`).

## Running Streamlit locally (dev)

If you prefer to run the Streamlit app outside Docker for faster iteration:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit/app.py
```

### Environment variables
- `AIRFLOW_API_BASE_URL` — Airflow REST API base (default: `http://airflow-webserver:8080/api/v1`)
- `AIRFLOW_UI_BASE_URL` — Airflow UI URL for links in the app
- `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` — credentials used by the Streamlit app to trigger DAGs (do not store in public repos)

Set these via a `.env` that your Compose file/containers load or export them in your shell for local runtimes.

## Deployment options

You can deploy this project in multiple ways depending on scale and budget. The two recommended options:

1) Single VM (Docker Compose) + Traefik (recommended for small teams)
- Run the whole stack on a Linux server using Docker Compose.
- Use Traefik as a reverse proxy for automatic TLS (Let's Encrypt).
- Good for small-to-medium workloads and keeps volumes and Airflow local to the host.

2) Cloud / Managed (Kubernetes or PaaS)
- Production-grade: deploy Airflow with an official Helm chart (Kubernetes), use managed Postgres, and run Streamlit/Next.js as services behind an Ingress + cert-manager.
- Use object storage (S3) for artifacts and backups.

### Important note about Vercel / Streamlit Cloud
Vercel is great to host a Next.js frontend, but it cannot run your stateful services (Airflow, Postgres, Redis) and cannot mount local volumes. If you later migrate the frontend to Next.js, host the backend on a VM or Kubernetes and point the frontend to secure API endpoints.

## Architecture & design notes

- Airflow orchestrates the end-to-end pipeline: extraction -> KPI normalization -> scoring -> writing outputs to `processed_data/`.
- Streamlit is a thin frontend that uploads files to `uploads/`, triggers the Airflow DAG, polls for results in `processed_data/`, and renders a dashboard with KPI tables and charts.
- ML models and tracking use `mlruns/` and `ml_models/` (MLflow integration).

## Troubleshooting

- Port conflicts: if `docker compose up` fails due to ports in use (e.g., 8501), identify and stop the local process (`lsof -nP -iTCP:8501 -sTCP:LISTEN`) or change the port mapping in `docker-compose.yaml`.
- Airflow init failures: re-run `docker compose up --build airflow-init` and inspect logs; ensure Postgres and Redis are healthy.
- Missing dependencies locally: install Python packages from `requirements.txt` (Streamlit, pandas, plotly, etc.).

## Contributing

Contributions are welcome. Suggested flow:

1. Fork the repo and create a feature branch
2. Run tests / lint locally (add tests where possible)
3. Open a pull request with a clear description

Please keep secrets out of PRs and use `.env` local files for credentials.

## What's next / roadmap

- Improve frontend UI (optional migration to Next.js for a richer dashboard)
- Add authenticated API endpoints for Streamlit to call (reduce direct Airflow exposure)
- Implement CI to build and publish Docker images and a deploy workflow

## License

This project is released under the MIT License. See `LICENSE` for details.

---

If you'd like, I can add a `docker-compose.traefik.yml`, update `docker-compose.yaml` with recommended Traefik labels, and add a `deploy/README.md` with exact commands.
# 🚀 Startup Health Score Dashboard  

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![Apache Airflow](https://img.shields.io/badge/Orchestration-Airflow-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)  
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)  
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

An **AI-powered platform** to automatically score startup health by analyzing business documents using a **Streamlit UI** and an orchestrated **Airflow pipeline**.  

---

## ✨ Features  

- 🔍 **Automated KPI Extraction** → Extracts **33+ KPIs** from PDFs, TXT, and images using an **OpenAI RAG model**  
- 📊 **Standardized Scoring** → Normalizes KPIs against **industry benchmarks** to generate an objective health score  
- 🤖 **ML-Powered Confidence** → A **RandomForest model** predicts a confidence score for each assessment’s reliability  
- ⚙️ **End-to-End Orchestration** → Automated with **Apache Airflow + Docker**  
- 🔄 **Continuous Learning** → Periodic retraining with **MLflow experiment tracking**  

---

## 🏗️ System Architecture  
![photo_2025-08-30_10-53-44](https://github.com/user-attachments/assets/04d3bfdb-a4d2-46ec-8d3b-8c5ffd1d3eee)


## 🛠️ Tech Stack

- **Frontend / UI** → Streamlit  
- **Pipeline Orchestration** → Apache Airflow  
- **Containerization** → Docker, Docker Compose  
- **AI / ML** → OpenAI, Scikit-learn, MLflow  
- **Databases** → PostgreSQL, Redis  

---

## ⚡ Quickstart

### ✅ Prerequisites
- Docker & Docker Compose installed  
- OpenAI API Key  

### 🚀 Setup & Run

Clone the repository:

```bash
git clone https://github.com/your-username/startup-health-engine.git
cd startup-health-engine
