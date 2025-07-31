import os
import sys
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.utils.dates import days_ago

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
BASE_DIR = '/opt/airflow'
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# --- IMPORTANT FIX: Add BASE_DIR to sys.path ---
# This ensures Python can find 'src' and 'config' modules when imported
# from within the DAG file, as they are relative to /opt/airflow.
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# --- END FIX ---

# Ensure processed_data directory exists (this will run inside the container)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Import your modules directly from src
# Ensure __init__.py files are correctly set up in src/core and src/utils
from src.core.config_loader import ConfigLoader
from src.core.kpi_extractor import KPIExtractor
from src.core.kpi_rag_extractor import KPIRAGExtractor
from src.core.scoring_engine import ScoringEngine
from src.utils.logger_config import logger, setup_logging

setup_logging() # Ensure logging is set up for DAG tasks


# --- Python Callable Functions for Tasks ---

def _check_for_documents(**kwargs):
    # Get the DAG run configuration
    dag_run = kwargs.get('dag_run')
    
    # --- NEW DEBUG LOGGING FOR DAG RUN CONF ---
    logger.info(f"DAG Run context received: {kwargs}")
    logger.info(f"DAG Run object: {dag_run}")
    if dag_run and dag_run.conf:
        logger.info(f"DAG Run conf received: {dag_run.conf}")
    else:
        logger.info("No DAG Run conf found in context.")
    # --- END NEW DEBUG LOGGING ---

    if dag_run and dag_run.conf and 'file_name' in dag_run.conf:
        file_name_from_conf = dag_run.conf['file_name']
        logger.info(f"Received file_name from DAG run config: {file_name_from_conf}")
        document_path = os.path.join(UPLOADS_DIR, file_name_from_conf)

        if not os.path.exists(document_path):
            logger.error(f"Configured document '{file_name_from_conf}' not found at {document_path}. Task will fail.")
            raise AirflowException(f"Document '{file_name_from_conf}' not found in 'uploads/' directory.")
        
        logger.info(f"Selected document from DAG conf: {file_name_from_conf} at {document_path}. Pushing to XCom.")
        kwargs['ti'].xcom_push(key='document_to_process_path', value=document_path)
    else:
        # Fallback to listing files if no file_name is provided in conf (e.g., manual trigger without conf)
        logger.warning("No 'file_name' found in DAG run configuration. Attempting to find any supported document in uploads directory.")
        ALLOWED_EXTENSIONS = ('.txt', '.pdf', '.jpg', '.jpeg', '.png')

        uploaded_files = [
            f for f in os.listdir(UPLOADS_DIR)
            if os.path.isfile(os.path.join(UPLOADS_DIR, f)) and \
               not f.startswith('.') and \
               f.lower().endswith(ALLOWED_EXTENSIONS)
        ]

        if not uploaded_files:
            logger.error(f"No supported documents found in the uploads directory: {UPLOADS_DIR}. Task will fail.")
            raise AirflowException(f"No supported documents found in the 'uploads/' directory. Please upload a .txt, .pdf, .jpg, .jpeg, or .png document.")

        selected_document_name = uploaded_files[0]
        document_path = os.path.join(UPLOADS_DIR, selected_document_name)

        logger.info(f"Found and selected document (fallback): {selected_document_name} at {document_path}. Pushing to XCom.")
        kwargs['ti'].xcom_push(key='document_to_process_path', value=document_path)


def _extract_raw_text(**kwargs):
    document_path = kwargs['ti'].xcom_pull(key='document_to_process_path')
    if not document_path:
        logger.error("No document path received from upstream task. Task will fail.")
        raise AirflowException("Document path not available from 'check_for_documents' task.")

    logger.info(f"Extracting raw text from {document_path}...")
    text_extractor = KPIExtractor()
    extracted_content = text_extractor.extract_text_from_document(document_path)

    if extracted_content:
        # Ensure the output file name is unique per run or based on input file
        output_file_name = os.path.basename(document_path).replace('.', '_') + "_extracted_text.txt"
        extracted_text_file_path = os.path.join(PROCESSED_DATA_DIR, output_file_name)
        try:
            with open(extracted_text_file_path, "w", encoding='utf-8') as f:
                f.write(extracted_content)
            logger.info(f"Extracted raw text saved to {extracted_text_file_path}")
            kwargs['ti'].xcom_push(key='raw_extracted_text', value=extracted_content)
        except IOError as e:
            logger.error(f"Failed to write extracted text to file {extracted_text_file_path}: {e}")
            raise AirflowException(f"Failed to write extracted text: {e}")
    else:
        logger.error(f"Raw text extraction failed or returned no content for {document_path}. Task will fail.")
        raise AirflowException("Raw text extraction returned no data.")


def _extract_kpis_rag(**kwargs):
    raw_text = kwargs['ti'].xcom_pull(key='raw_extracted_text')
    if not raw_text:
        logger.error("No raw text received from upstream task. Task will fail.")
        raise AirflowException("Raw text not available from 'extract_raw_text' task.")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Cannot extract KPIs.")
        raise AirflowException("OPENAI_API_KEY is not set. Please configure it in your Docker Compose environment or .env file.")

    logger.info("Loading KPI benchmarks and initializing RAG extractor...")
    try:
        config_loader = ConfigLoader(config_dir=CONFIG_DIR)
        kpi_benchmarks = config_loader.load_config(config_loader.kpi_benchmarks_path)
        kpi_benchmark_map = config_loader.get_kpi_benchmark_map(kpi_benchmarks)

        rag_extractor = KPIRAGExtractor(kpi_benchmarks, openai_api_key=openai_api_key, kpi_benchmark_map=kpi_benchmark_map)
        logger.info("Extracting KPIs using RAG (LLM)...")
        kpi_dict, llm_response = rag_extractor.extract_kpis(raw_text)

        if kpi_dict:
            # Ensure the output file name is unique per run or based on input file
            # Assuming document_path was pushed by _check_for_documents
            document_path_from_xcom = kwargs['ti'].xcom_pull(key='document_to_process_path')
            output_file_name = os.path.basename(document_path_from_xcom).replace('.', '_') + "_extracted_kpis.json"
            extracted_kpi_file_path = os.path.join(PROCESSED_DATA_DIR, output_file_name)

            with open(extracted_kpi_file_path, 'w', encoding='utf-8') as f:
                json.dump(kpi_dict, f, indent=2)
            logger.info(f"Extracted KPIs via RAG saved to {extracted_kpi_file_path}")
            kwargs['ti'].xcom_push(key='extracted_kpi_dict', value=kpi_dict)
        else:
            logger.error(f"KPI extraction via RAG failed or returned no data. LLM Response: {llm_response}. Task will fail.")
            raise AirflowException(f"KPI extraction via RAG returned no data. LLM Response: {llm_response}")
    except Exception as e:
        logger.error(f"An error occurred during KPI extraction: {e}", exc_info=True)
        raise AirflowException(f"KPI extraction failed: {e}")


def _calculate_scores(**kwargs):
    logger.info("Calculating startup health scores...")
    kpi_input_data = kwargs['ti'].xcom_pull(key='extracted_kpi_dict')
    if not kpi_input_data:
        logger.warning(f"KPI dict not found in XCom. Attempting to load from file: {PROCESSED_DATA_DIR}/*.json") # Adjusting warning
        # Try to find the latest extracted_kpis.json if not in XCom (fallback for manual debugging)
        latest_kpi_file = None
        for f_name in os.listdir(PROCESSED_DATA_DIR):
            if f_name.endswith("_extracted_kpis.json"):
                f_path = os.path.join(PROCESSED_DATA_DIR, f_name)
                if not latest_kpi_file or os.path.getmtime(f_path) > os.path.getmtime(latest_kpi_file):
                    latest_kpi_file = f_path
        
        if not latest_kpi_file:
            logger.error(f"No extracted KPI file found in {PROCESSED_DATA_DIR}. Task will fail.")
            raise AirflowException(f"No extracted KPI file found. Ensure previous task succeeded.")
        
        try:
            with open(latest_kpi_file, 'r', encoding='utf-8') as f:
                kpi_input_data = json.load(f)
            logger.info(f"Loaded KPI input from {latest_kpi_file}")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read or parse extracted KPI file {latest_kpi_file}: {e}")
            raise AirflowException(f"Failed to load extracted KPIs: {e}")

    # Get OpenAI API key from environment for ScoringEngine
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY environment variable not set. LLM-based suggestions in ScoringEngine will be skipped.")

    try:
        config_loader = ConfigLoader(config_dir=CONFIG_DIR)
        # Pass the API key to ScoringEngine
        scoring_engine = ScoringEngine(config_loader=config_loader, openai_api_key=openai_api_key)

        results = scoring_engine.calculate_scores(kpi_input_data)

        # Ensure the output file name is unique per run or based on input file
        document_path_from_xcom = kwargs['ti'].xcom_pull(key='document_to_process_path')
        output_file_name_prefix = os.path.basename(document_path_from_xcom).replace('.', '_') if document_path_from_xcom else "unknown_document"
        score_output_file_path = os.path.join(PROCESSED_DATA_DIR, f"{output_file_name_prefix}_startup_score_output.json")

        with open(score_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Scoring results saved to {score_output_file_path}")

        if "error" in results: # Check for a general error key, though specific errors are now handled by AirflowException
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