import os
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowException

# import sys # Removed sys.path manipulation

# --- Configuration for DAG ---
DAG_ID = 'startup_health_score_full_pipeline'
DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define paths relative to the container's /opt/airflow directory
# Airflow will now find modules in /opt/airflow/src because PYTHONPATH is set
BASE_DIR = '/opt/airflow' # This is where your project root is mounted inside the container
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
SAMPLE_DOCUMENT_NAME = 'sample_document.txt' # Example: Use your PDF here for testing the full flow
EXTRACTED_TEXT_FILE = os.path.join(PROCESSED_DATA_DIR, 'extracted_document_content.txt')
EXTRACTED_KPI_FILE = os.path.join(PROCESSED_DATA_DIR, 'extracted_kpis.json')
SCORE_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'startup_score_output.json')

# Ensure processed_data directory exists (this will run inside the container)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Now import your modules directly from src
# Make sure your __init__.py files are correctly set up in src/core and src/utils
from src.core.config_loader import ConfigLoader
from src.core.kpi_extractor import KPIExtractor
from src.core.kpi_rag_extractor import KPIRAGExtractor
from src.core.scoring_engine import ScoringEngine
from src.utils.logger_config import logger, setup_logging

setup_logging() # Ensure logging is set up for DAG tasks


# --- Python Callable Functions for Tasks ---

def _check_for_documents(**kwargs):
    document_path = os.path.join(UPLOADS_DIR, SAMPLE_DOCUMENT_NAME)
    if not os.path.exists(document_path):
        logger.error(f"No document found at {document_path}. Task will fail.")
        raise AirflowException(f"Document {SAMPLE_DOCUMENT_NAME} not found. Please ensure it's in the 'uploads/' directory.")
    logger.info(f"Document {SAMPLE_DOCUMENT_NAME} found. Proceeding.")
    kwargs['ti'].xcom_push(key='document_to_process_path', value=document_path)


def _extract_raw_text(**kwargs):
    document_path = kwargs['ti'].xcom_pull(key='document_to_process_path')
    if not document_path:
        logger.error("No document path received from upstream task. Task will fail.")
        raise AirflowException("Document path not available from 'check_for_documents' task.")

    text_extractor = KPIExtractor()
    extracted_content = text_extractor.extract_text_from_document(document_path)

    if extracted_content:
        try:
            with open(EXTRACTED_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(extracted_content)
            logger.info(f"Extracted raw text saved to {EXTRACTED_TEXT_FILE}")
            kwargs['ti'].xcom_push(key='raw_extracted_text', value=extracted_content) # Push raw text for next task
        except IOError as e:
            logger.error(f"Failed to write extracted text to file {EXTRACTED_TEXT_FILE}: {e}")
            raise AirflowException(f"Failed to write extracted text: {e}")
    else:
        logger.error(f"Raw text extraction failed or returned no content for {document_path}. Task will fail.")
        raise AirflowException("Raw text extraction returned no data.")


def _extract_kpis_rag(**kwargs):
    raw_text = kwargs['ti'].xcom_pull(key='raw_extracted_text')
    if not raw_text:
        logger.error("No raw text received from upstream task. Task will fail.")
        raise AirflowException("Raw text not available from 'extract_raw_text' task.")

    config_loader = ConfigLoader(config_dir=CONFIG_DIR)
    kpi_benchmarks = config_loader.load_config(config_loader.kpi_benchmarks_path)

    rag_extractor = KPIRAGExtractor(kpi_benchmarks)
    kpi_dict = rag_extractor.extract_kpis(raw_text)

    if kpi_dict:
        try:
            with open(EXTRACTED_KPI_FILE, 'w', encoding='utf-8') as f:
                json.dump(kpi_dict, f, indent=2)
            logger.info(f"Extracted KPIs via RAG saved to {EXTRACTED_KPI_FILE}")
            kwargs['ti'].xcom_push(key='extracted_kpi_dict', value=kpi_dict) # Push kpi_dict for next task
        except IOError as e:
            logger.error(f"Failed to write extracted KPIs to file {EXTRACTED_KPI_FILE}: {e}")
            raise AirflowException(f"Failed to write extracted KPIs: {e}")
    else:
        logger.error("KPI extraction via RAG failed or returned no data. Task will fail.")
        raise AirflowException("KPI extraction via RAG returned no data.")


def _calculate_scores(**kwargs):
    # Prefer pulling kpi_dict from XCom if available, fallback to file
    kpi_input_data = kwargs['ti'].xcom_pull(key='extracted_kpi_dict')
    if not kpi_input_data:
        logger.warning(f"KPI dict not found in XCom. Attempting to load from file: {EXTRACTED_KPI_FILE}")
        if not os.path.exists(EXTRACTED_KPI_FILE):
            logger.error(f"Extracted KPI file not found: {EXTRACTED_KPI_FILE}. Task will fail.")
            raise AirflowException(f"Extracted KPI file {EXTRACTED_KPI_FILE} not found. Ensure previous task succeeded.")
        try:
            with open(EXTRACTED_KPI_FILE, 'r', encoding='utf-8') as f:
                kpi_input_data = json.load(f)
            logger.info(f"Loaded KPI input from {EXTRACTED_KPI_FILE}")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read or parse extracted KPI file {EXTRACTED_KPI_FILE}: {e}")
            raise AirflowException(f"Failed to load extracted KPIs: {e}")

    try:
        config_loader = ConfigLoader(config_dir=CONFIG_DIR)
        scoring_engine = ScoringEngine(config_loader=config_loader)

        results = scoring_engine.calculate_scores(kpi_input_data)

        with open(SCORE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Scoring results saved to {SCORE_OUTPUT_FILE}")

        if "error" in results:
            logger.error(f"Scoring completed with errors: {results['error']}")
            raise AirflowException(f"Scoring failed: {results['error']}")
        else:
            logger.info(f"Total Score: {results.get('total_score', 'N/A')}")
    except Exception as e:
        logger.error(f"An error occurred during score calculation: {e}", exc_info=True)
        raise AirflowException(f"Score calculation failed: {e}")


# --- DAG Definition ---
with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description='A full pipeline to extract text, extract KPIs using RAG, and calculate startup health scores.',
    schedule_interval=None, # Set to None for manual trigger, or use a cron expression for scheduled runs
    start_date=days_ago(1),
    tags=['startup', 'kpi', 'scoring', 'data_pipeline', 'llm'],
    catchup=False,
) as dag:
    # Task 1: Check for new documents
    check_documents_task = PythonOperator(
        task_id='check_for_documents',
        python_callable=_check_for_documents,
        provide_context=True,
    )

    # Task 2: Extract raw text from the document
    extract_raw_text_task = PythonOperator(
        task_id='extract_raw_text_from_document',
        python_callable=_extract_raw_text,
        provide_context=True,
    )

    # Task 3: Extract KPIs using RAG (LLM)
    extract_kpis_rag_task = PythonOperator(
        task_id='extract_kpis_using_rag',
        python_callable=_extract_kpis_rag,
        provide_context=True,
    )

    # Task 4: Calculate Health Scores
    calculate_scores_task = PythonOperator(
        task_id='calculate_startup_health_scores',
        python_callable=_calculate_scores,
        provide_context=True,
    )

    # Define task dependencies
    check_documents_task >> extract_raw_text_task >> extract_kpis_rag_task >> calculate_scores_task
