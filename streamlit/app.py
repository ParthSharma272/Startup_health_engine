import streamlit as st
import os
import json
import time
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import logging
import re
import pandas as pd # Added for better data handling for charts

# Configure Streamlit page
# Removed 'icon' argument again due to persistent TypeError.
st.set_page_config(layout="wide", page_title="Startup Health Score Dashboard 🚀")

# Define paths relative to the Streamlit app's container WORKDIR (/app)
UPLOADS_DIR = './uploads'
PROCESSED_DATA_DIR = './processed_data'

# Ensure necessary directories exist within the Streamlit container
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Airflow API Configuration
AIRFLOW_API_BASE_URL = os.environ.get("AIRFLOW_API_BASE_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_UI_BASE_URL = os.environ.get("AIRFLOW_UI_BASE_URL", "http://localhost:8080")
AIRFLOW_DAG_ID = 'startup_health_score_full_pipeline'

# Airflow API Credentials (for local testing with default admin user)
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")

# Basic logging for Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

# --- Helper Functions ---
def trigger_airflow_dag(dag_id: str, file_name: str) -> dict:
    """Triggers an Airflow DAG run via the REST API."""
    
    # Sanitize file_name for dag_run_id: replace non-alphanumeric, non-underscore, non-dot, non-tilde, non-colon, non-plus, non-hyphen with underscore
    # This matches Airflow's allowed pattern: '^[A-Za-z0-9_.~:+-]+$'
    sanitized_file_name = re.sub(r'[^A-Za-z0-9_.~:+-]', '_', file_name)
    
    dag_run_id = f"streamlit_trigger_{sanitized_file_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    endpoint = f"{AIRFLOW_API_BASE_URL}/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": dag_run_id, # Use the sanitized ID
        "conf": {"file_name": file_name} # Pass original file name in conf
    }
    headers = {"Content-Type": "application/json"}
    auth = HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)

    app_logger.info(f"Attempting to trigger DAG {dag_id} with run ID '{dag_run_id}' at {endpoint}")
    try:
        response = requests.post(endpoint, json=payload, headers=headers, auth=auth, timeout=10)
        response.raise_for_status()
        app_logger.info(f"Successfully triggered DAG {dag_id}. Response: {response.json()}")
        return response.json()
    except requests.exceptions.ConnectionError as e:
        app_logger.error(f"Connection error to Airflow API: {e}. Is Airflow webserver running and accessible?")
        st.error(f"Connection Error: Could not connect to Airflow API. Is the Airflow webserver running? ({AIRFLOW_API_BASE_URL})")
        return {}
    except requests.exceptions.Timeout:
        app_logger.error("Timeout connecting to Airflow API.")
        st.error("Timeout: Airflow API did not respond in time.")
        return {}
    except requests.exceptions.RequestException as e:
        app_logger.error(f"Failed to trigger DAG {dag_id}. Error: {e}")
        st.error(f"Failed to trigger Airflow DAG. Error: {e}")
        return {}

def check_for_output_files():
    """
    Checks if both required output files (_extracted_kpis.json and _startup_score_output.json)
    exist in the processed_data directory.
    """
    files_in_processed_data = os.listdir(PROCESSED_DATA_DIR)
    app_logger.info(f"Checking for output files in {PROCESSED_DATA_DIR}. Current files: {files_in_processed_data}")
    
    kpi_file_found = any(f.endswith("_extracted_kpis.json") for f in files_in_processed_data)
    score_file_found = any(f.endswith("_startup_score_output.json") for f in files_in_processed_data)
    
    return kpi_file_found and score_file_found


def load_output_data():
    """Loads KPI and score data from processed_data directory."""
    kpis = {}
    scores = {}
    
    app_logger.info(f"Attempting to load output data from {PROCESSED_DATA_DIR}")
    files_in_processed_data = os.listdir(PROCESSED_DATA_DIR)
    app_logger.info(f"Files found in {PROCESSED_DATA_DIR}: {files_in_processed_data}")

    # Find the most recent output files
    latest_kpi_file = None
    latest_score_file = None
    
    for f_name in files_in_processed_data:
        f_path = os.path.join(PROCESSED_DATA_DIR, f_name)
        if f_name.endswith("_extracted_kpis.json"):
            if not latest_kpi_file or os.path.getmtime(f_path) > os.path.getmtime(latest_kpi_file):
                latest_kpi_file = f_path
        elif f_name.endswith("_startup_score_output.json"):
            if not latest_score_file or os.path.getmtime(f_path) > os.path.getmtime(latest_score_file):
                latest_score_file = f_path

    app_logger.info(f"Latest KPI file found: {latest_kpi_file}")
    app_logger.info(f"Latest Score file found: {latest_score_file}")

    if not latest_kpi_file or not latest_score_file:
        app_logger.error("Output files not found. Airflow DAG might not have completed or failed.")
        st.error("Analysis results not found. The Airflow pipeline might not have completed successfully.")
        return {}, {} # Return empty if files are missing

    try:
        with open(latest_kpi_file, 'r', encoding='utf-8') as f:
            kpis = json.load(f)
        with open(latest_score_file, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        app_logger.info("Successfully loaded KPI and score output files.")
    except json.JSONDecodeError as e:
        app_logger.error(f"Error decoding JSON from output files: {e}")
        st.error(f"Error reading analysis results (invalid JSON): {e}")
    except Exception as e:
        app_logger.error(f"Error loading output data: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while loading results: {e}")
    return kpis, scores

# --- Streamlit UI ---
st.title("🚀 Startup Health Score Dashboard")
st.markdown("""
Welcome! Upload a business document (TXT, JPG, PNG, PDF) to get an automated health score.
Our intelligent pipeline, powered by Airflow, will extract key performance indicators (KPIs)
using advanced AI and calculate a comprehensive health score based on industry benchmarks.
""")

# Sidebar for controls and info
with st.sidebar:
    st.header("About This App")
    st.info(f"""
        This dashboard is the frontend for a robust Airflow-orchestrated data pipeline.
        
        1.  **Upload:** Your document is saved to a shared volume.
        2.  **Trigger:** An Airflow DAG is triggered via its REST API.
        3.  **Process:** Airflow handles text extraction, AI-powered KPI extraction, and scoring.
        4.  **Display:** Results are written to another shared volume and displayed here.
        
        [Go to Airflow UI]({AIRFLOW_UI_BASE_URL})
    """)
    st.markdown("---")
    if st.button("🔄 Reset Application State", help="Clear all uploaded files and analysis results."):
        # Clear all files in uploads and processed_data
        for directory in [UPLOADS_DIR, PROCESSED_DATA_DIR]:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    app_logger.info(f"Deleted: {item_path}")
        
        st.session_state.clear()
        # Ensure default values are set after clearing for a clean restart
        st.session_state['processing_stage'] = 'initial'
        st.session_state['uploaded_file_name'] = None
        st.session_state['dag_run_id'] = None
        st.session_state['extracted_kpis_display'] = None
        st.session_state['final_scores_display'] = None
        st.success("App state and uploaded files reset!")
        time.sleep(1) # Give user time to see message
        st.rerun()

# --- Main Content Area ---
st.markdown("---")
st.header("1. Upload Your Document 📄")

# Initialize session state variables if they don't exist
if 'processing_stage' not in st.session_state:
    st.session_state['processing_stage'] = 'initial'
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None
if 'dag_run_id' not in st.session_state:
    st.session_state['dag_run_id'] = None
if 'extracted_kpis_display' not in st.session_state:
    st.session_state['extracted_kpis_display'] = None
if 'final_scores_display' not in st.session_state:
    st.session_state['final_scores_display'] = None
if 'last_processed_file_path' not in st.session_state:
    st.session_state['last_processed_file_path'] = None


uploaded_file = st.file_uploader(
    "Drag and drop your document here or click to browse",
    type=["txt", "jpg", "jpeg", "png", "pdf"],
    key="file_uploader"
)

# Handle file upload and saving
if uploaded_file is not None:
    # Only process if a new file is uploaded or the name changes
    if st.session_state['uploaded_file_name'] != uploaded_file.name:
        st.session_state['uploaded_file_name'] = uploaded_file.name
        st.session_state['processing_stage'] = 'uploaded' # Reset stage for new file
        st.session_state['dag_run_id'] = None # Clear previous run ID
        st.session_state['extracted_kpis_display'] = None
        st.session_state['final_scores_display'] = None

        file_path_in_uploads = os.path.join(UPLOADS_DIR, uploaded_file.name)
        try:
            # Clear previous files in uploads to ensure only one is processed
            for item in os.listdir(UPLOADS_DIR):
                item_path = os.path.join(UPLOADS_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    app_logger.info(f"Deleted old uploaded file: {item_path}")

            with open(file_path_in_uploads, "wb") as f:
                f.write(uploaded_file.getbuffer())
            app_logger.info(f"File '{uploaded_file.name}' saved to {UPLOADS_DIR}")
            st.success(f"✅ Document **'{uploaded_file.name}'** uploaded successfully to shared volume.")
            st.session_state['last_processed_file_path'] = file_path_in_uploads
        except Exception as e:
            st.error(f"❌ Error saving file: {e}")
            st.session_state['processing_stage'] = 'error'
            app_logger.error(f"Error saving uploaded file: {e}", exc_info=True)
        # No st.rerun() here, let Streamlit re-render naturally

    # Display current uploaded file info if available
    if st.session_state.get('uploaded_file_name'):
        st.info(f"Current document selected: **{st.session_state['uploaded_file_name']}**")

        # --- Trigger Airflow Processing Button ---
        st.markdown("---")
        st.header("2. Start Analysis 🚀")
        st.write("Click the button below to send your document for processing by the Airflow pipeline.")
        
        # Only show the button if not already triggered/processing
        if st.session_state['processing_stage'] in ['uploaded', 'error', 'completed']:
            if st.button("🚀 Start Analysis via Airflow", help="Initiate the data pipeline in Airflow."):
                # Clear previous processed data before starting a new run
                for item in os.listdir(PROCESSED_DATA_DIR):
                    item_path = os.path.join(PROCESSED_DATA_DIR, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        app_logger.info(f"Deleted old processed file: {item_path}")

                with st.spinner("Initiating Airflow DAG..."):
                    dag_trigger_response = trigger_airflow_dag(AIRFLOW_DAG_ID, st.session_state['uploaded_file_name'])
                    if dag_trigger_response and 'dag_run_id' in dag_trigger_response:
                        st.session_state['dag_run_id'] = dag_trigger_response['dag_run_id']
                        st.session_state['processing_stage'] = 'triggered'
                        st.success(f"✅ Airflow DAG triggered! DAG Run ID: `{st.session_state['dag_run_id']}`")
                        st.info(f"Monitoring progress... This may take a few minutes. [View in Airflow UI]({AIRFLOW_UI_BASE_URL}/dags/{AIRFLOW_DAG_ID}/grid?dag_run_id={st.session_state['dag_run_id']})")
                        time.sleep(2) # Give a moment for message to show
                        st.rerun() # Rerun to move to processing stage
                    else:
                        st.session_state['processing_stage'] = 'error'
                        st.error("❌ Failed to trigger Airflow DAG. Check console logs for details.")
        elif st.session_state['processing_stage'] == 'triggered':
            st.info("Analysis already triggered. Monitoring status below.")


    # --- Monitor and Display Results ---
    if st.session_state['processing_stage'] in ['triggered', 'processing']:
        st.markdown("---")
        st.header("3. Processing Status ⏳")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        dag_run_id = st.session_state.get('dag_run_id')
        if dag_run_id:
            status_message = status_placeholder.info(f"⏳ Airflow DAG run `{dag_run_id}` is in progress. Checking for results...")
            
            max_retries = 180 # Increased to 30 minutes (180 * 10 seconds)
            retries = 0
            
            while not check_for_output_files() and retries < max_retries:
                progress_bar.progress((retries + 1) / max_retries)
                time.sleep(10) # Wait for 10 seconds before checking again
                retries += 1
                status_placeholder.info(f"⏳ Airflow DAG run `{dag_run_id}` is in progress. Checking for results... (Attempt {retries}/{max_retries})")
                app_logger.info(f"Polling for output files... Attempt {retries}/{max_retries}")

            if check_for_output_files():
                st.session_state['processing_stage'] = 'completed'
                status_placeholder.empty()
                progress_bar.empty()
                st.success("🎉 Airflow processing complete! Results are ready.")
                st.balloons() # Add a celebratory animation
                st.rerun()
            else:
                st.session_state['processing_stage'] = 'error'
                status_placeholder.empty()
                progress_bar.empty()
                st.error(f"❌ Airflow processing timed out or output files not found. Please check the [Airflow UI]({AIRFLOW_UI_BASE_URL}) for detailed logs.")
                app_logger.error("Airflow processing timed out or output files not found.")

    if st.session_state['processing_stage'] == 'completed':
        st.markdown("---")
        st.header("4. Analysis Results ✨")
        
        # Load and display results
        extracted_kpis, final_scores = load_output_data()
        
        if extracted_kpis and final_scores:
            st.session_state['extracted_kpis_display'] = extracted_kpis
            st.session_state['final_scores_display'] = final_scores

            # Layout for overall score and download button
            col1, col2 = st.columns([2, 1])
            with col1:
                total_score = st.session_state['final_scores_display'].get('total_score', 0.0)
                st.markdown(f"""
                    <div style="
                        background-color: #262730; 
                        padding: 20px; 
                        border-radius: 15px; 
                        text-align: center; 
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                        margin-bottom: 20px;
                    ">
                        <h3 style="color: #ADD8E6; margin-bottom: 10px;">Overall Startup Health Score</h3>
                        <p style="
                            font-size: 3.5em; 
                            font-weight: bold; 
                            color: #4CAF50; 
                            line-height: 1;
                        ">{total_score:.2f} <span style="font-size: 0.5em;">/ 100</span></p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True) # Add some vertical space
                results_json = json.dumps(st.session_state['final_scores_display'], indent=2)
                st.download_button(
                    label="📥 Download Full Analysis Results (JSON)",
                    data=results_json,
                    file_name=f"startup_health_score_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                    mime="application/json",
                    help="Download the complete KPI and score data as a JSON file."
                )
            st.markdown("---")


            # Display Category Scores in a visually appealing chart and table
            st.subheader("📊 Category Scores")
            category_scores_data = []
            for category, score in st.session_state['final_scores_display']['category_scores'].items():
                category_scores_data.append({"Category": category.replace('_', ' ').title(), "Score": score}) # Keep score as float for chart
            
            if category_scores_data:
                df_category_scores = pd.DataFrame(category_scores_data)
                # Sort for consistent chart display
                df_category_scores = df_category_scores.sort_values(by="Score", ascending=False) 

                st.bar_chart(df_category_scores.set_index("Category"), use_container_width=True, height=300)
                
                with st.expander("View Raw Category Scores Table"):
                    # Format for display in table
                    df_category_scores['Score'] = df_category_scores['Score'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(df_category_scores, use_container_width=True, hide_index=True)
            else:
                st.info("No category scores available to display.")


            st.markdown("---")

            # Display Extracted KPIs in an expander
            with st.expander("📋 View Extracted KPIs", expanded=False):
                kpi_data_for_df = [{"KPI": k, "Value": v} for k, v in st.session_state['extracted_kpis_display'].items()]
                st.dataframe(kpi_data_for_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Display Normalized KPI Scores in an expander
            with st.expander("📈 View Normalized KPI Scores", expanded=False):
                normalized_kpi_data_for_df = [{"KPI": k, "Normalized Score": f"{v:.2f}"} for k, v in st.session_state['final_scores_display']['normalized_kpis'].items()]
                st.dataframe(normalized_kpi_data_for_df, use_container_width=True, hide_index=True)
            
            # Display missing KPIs
            if st.session_state['final_scores_display'].get('missing_mandatory_kpis'):
                st.warning(f"⚠️ Missing Mandatory KPIs: {', '.join(st.session_state['final_scores_display']['missing_mandatory_kpis'])}. Score might be impacted.")
            if st.session_state['final_scores_display'].get('missing_non_mandatory_kpis'):
                st.info(f"ℹ️ Missing Non-Mandatory KPIs: {', '.join(st.session_state['final_scores_display']['missing_non_mandatory_kpis'])}. Consider including these for a more comprehensive score.")

        else:
            st.error("❌ Could not load processed data. Please check the 'processed_data' directory and Airflow logs.")

    elif st.session_state['processing_stage'] == 'error':
        st.error(f"❌ An error occurred during processing. Please check the [Airflow UI]({AIRFLOW_UI_BASE_URL}) for detailed logs.")

else:
    st.info("⬆️ Upload a document to begin the analysis.")

