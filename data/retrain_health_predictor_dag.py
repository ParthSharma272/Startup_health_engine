# dags/retrain_health_predictor_dag.py
import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Add BASE_DIR to sys.path
BASE_DIR = '/opt/airflow'
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.ml_models.health_predictor import HealthPredictor
from src.utils.logger_config import logger, setup_logging

setup_logging()

# DAG Configuration
DAG_ID = 'retrain_health_predictor_model'
DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define paths
MODEL_DIR = '/opt/airflow/ml_models'
LOG_FILE_PATH = os.path.join(MODEL_DIR, 'health_scores_log.csv')
MIN_SAMPLES_FOR_TRAINING = 50  # Minimum samples required for training

def _check_retraining_requirements(**kwargs):
    """Check if there's enough data for retraining."""
    if not os.path.exists(LOG_FILE_PATH):
        logger.info("No training data log file found. Skipping retraining.")
        kwargs['ti'].xcom_push(key='should_retrain', value=False)
        return
    
    try:
        # Count lines in log file (subtract 1 for header)
        with open(LOG_FILE_PATH, 'r') as f:
            line_count = sum(1 for _ in f) - 1
        
        logger.info(f"Found {line_count} samples in training data log.")
        
        if line_count >= MIN_SAMPLES_FOR_TRAINING:
            logger.info("Sufficient data available for retraining.")
            kwargs['ti'].xcom_push(key='should_retrain', value=True)
            kwargs['ti'].xcom_push(key='sample_count', value=line_count)
        else:
            logger.info(f"Insufficient data for retraining. Need at least {MIN_SAMPLES_FOR_TRAINING} samples.")
            kwargs['ti'].xcom_push(key='should_retrain', value=False)
    except Exception as e:
        logger.error(f"Error checking retraining requirements: {str(e)}")
        kwargs['ti'].xcom_push(key='should_retrain', value=False)

def _retrain_model(**kwargs):
    """Retrain the health predictor model."""
    should_retrain = kwargs['ti'].xcom_pull(key='should_retrain')
    
    if not should_retrain:
        logger.info("Skipping model retraining as requirements not met.")
        return
    
    sample_count = kwargs['ti'].xcom_pull(key='sample_count', default=0)
    logger.info(f"Starting model retraining with {sample_count} samples...")
    
    try:
        # Initialize health predictor
        health_predictor = HealthPredictor(model_dir=MODEL_DIR)
        
        # Train model
        training_results = health_predictor.train_model(LOG_FILE_PATH)
        
        if training_results['status'] == 'success':
            logger.info("Model retraining completed successfully.")
            logger.info(f"Training metrics: {training_results}")
            
            # Update Airflow variable with last training timestamp
            Variable.set("health_predictor_last_trained", datetime.now().isoformat())
            
            # Push results to XCom
            kwargs['ti'].xcom_push(key='training_results', value=training_results)
        else:
            logger.error(f"Model retraining failed: {training_results.get('message', 'Unknown error')}")
            raise AirflowException(f"Model retraining failed: {training_results.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}", exc_info=True)
        raise AirflowException(f"Error during model retraining: {str(e)}")

def _archive_training_data(**kwargs):
    """Archive training data after successful retraining."""
    should_retrain = kwargs['ti'].xcom_pull(key='should_retrain')
    
    if not should_retrain:
        logger.info("Skipping data archiving as retraining was not performed.")
        return
    
    try:
        # Create archive directory if it doesn't exist
        archive_dir = os.path.join(MODEL_DIR, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        
        # Archive log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(archive_dir, f"health_scores_log_{timestamp}.csv")
        
        # Copy current log to archive
        import shutil
        shutil.copy2(LOG_FILE_PATH, archive_path)
        
        # Clear current log file (keep header)
        with open(LOG_FILE_PATH, 'w') as f:
            f.write("timestamp,document_name,total_score,confidence_score,prediction_method," +
                    "normalized_kpis,category_scores,missing_mandatory_kpis,missing_non_mandatory_kpis\n")
        
        logger.info(f"Training data archived to {archive_path}")
    except Exception as e:
        logger.error(f"Error archiving training data: {str(e)}", exc_info=True)
        # Don't fail the DAG for archiving errors

# DAG Definition
with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description='Retrain the health predictor model using accumulated data',
    schedule_interval=timedelta(weeks=1),  # Run weekly
    start_date=days_ago(1),
    tags=['ml', 'retraining', 'health_predictor'],
    catchup=False,
    max_active_runs=1,
) as dag:
    
    check_requirements_task = PythonOperator(
        task_id='check_retraining_requirements',
        python_callable=_check_retraining_requirements,
        provide_context=True,
    )
    
    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=_retrain_model,
        provide_context=True,
    )
    
    archive_data_task = PythonOperator(
        task_id='archive_training_data',
        python_callable=_archive_training_data,
        provide_context=True,
    )
    
    # Define task dependencies
    check_requirements_task >> retrain_model_task >> archive_data_task