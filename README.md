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
