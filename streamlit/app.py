import streamlit as st
import os
import json
import time
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import logging
import re
import pandas as pd

# Configure Streamlit page
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
    
    sanitized_file_name = re.sub(r'[^A-Za-z0-9_.~:+-]', '_', file_name)
    dag_run_id = f"streamlit_trigger_{sanitized_file_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    endpoint = f"{AIRFLOW_API_BASE_URL}/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": dag_run_id,
        "conf": {"file_name": file_name}
    }
    headers = {"Content-Type": "application/json"}

    app_logger.info(f"Attempting to trigger DAG {dag_id} with run ID '{dag_run_id}' at {endpoint}")
    try:
        # Use basic auth since authentication can't be fully disabled
        auth = HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
        response = requests.post(endpoint, json=payload, headers=headers, auth=auth, timeout=10)
        response.raise_for_status()
        app_logger.info(f"Successfully triggered DAG {dag_id}. Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        app_logger.error(f"Failed to trigger DAG {dag_id}. Error: {e}")
        st.error(f"Failed to trigger Airflow DAG. Error: {e}")
        return {}

def check_for_specific_output_files(file_name_prefix: str) -> bool:
    """
    Checks if both required output files (_extracted_kpis.json and _startup_score_output.json)
    for a specific document exist in the processed_data directory.
    """
    files_in_processed_data = os.listdir(PROCESSED_DATA_DIR)
    app_logger.info(f"Checking for output files for prefix '{file_name_prefix}' in {PROCESSED_DATA_DIR}. Current files: {files_in_processed_data}")
    
    kpi_file_expected = f"{file_name_prefix}_extracted_kpis.json"
    score_file_expected = f"{file_name_prefix}_startup_score_output.json"
    
    kpi_file_found = kpi_file_expected in files_in_processed_data
    score_file_found = score_file_expected in files_in_processed_data
    
    return kpi_file_found and score_file_found

def load_output_data_for_document(file_name_prefix: str) -> tuple[dict, dict]:
    """Loads KPI and score data for a specific document from processed_data directory."""
    kpis = {}
    scores = {}
    
    app_logger.info(f"Attempting to load output data for '{file_name_prefix}' from {PROCESSED_DATA_DIR}")

    kpi_file_path = os.path.join(PROCESSED_DATA_DIR, f"{file_name_prefix}_extracted_kpis.json")
    score_file_path = os.path.join(PROCESSED_DATA_DIR, f"{file_name_prefix}_startup_score_output.json")

    if not os.path.exists(kpi_file_path) or not os.path.exists(score_file_path):
        app_logger.error(f"Output files for '{file_name_prefix}' not found. KPI: {os.path.exists(kpi_file_path)}, Score: {os.path.exists(score_file_path)}")
        st.error(f"Analysis results for '{file_name_prefix}' not found. The Airflow pipeline might not have completed or failed to write these specific files.")
        return {}, {}

    try:
        with open(kpi_file_path, 'r', encoding='utf-8') as f:
            kpis = json.load(f)
        with open(score_file_path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        app_logger.info(f"Successfully loaded KPI and score output files for '{file_name_prefix}'.")
    except json.JSONDecodeError as e:
        app_logger.error(f"Error decoding JSON from output files for '{file_name_prefix}': {e}")
        st.error(f"Error reading analysis results for '{file_name_prefix}' (invalid JSON): {e}")
    except Exception as e:
        app_logger.error(f"Error loading output data for '{file_name_prefix}': {e}", exc_info=True)
        st.error(f"An unexpected error occurred while loading results for '{file_name_prefix}': {e}")
    return kpis, scores

def get_file_prefix(file_name: str) -> str:
    """Generates the file prefix used for processed output files."""
    # Ensure this matches the DAG's file naming convention
    return file_name.replace('.', '_')

# --- Streamlit UI ---
st.title("🚀 Startup Health Score Dashboard")
st.markdown("""
Welcome! Upload business documents (TXT, JPG, PNG, PDF) to get automated health scores.
Our intelligent pipeline, powered by Airflow, will extract key performance indicators (KPIs)
using advanced AI and calculate a comprehensive health score based on industry benchmarks.
""")

# Sidebar for controls and info
with st.sidebar:
    st.header("About This App")
    st.info(f"""
        This dashboard is the frontend for a robust Airflow-orchestrated data pipeline.
        
        1.  **Upload:** Your document is saved to a shared volume.
        2.  **Trigger:** An Airflow DAG is triggered via its REST API for each document.
        3.  **Process:** Airflow handles text extraction, AI-powered KPI extraction, and scoring.
        4.  **Display:** Results are written to another shared volume and displayed here.
        
        [Go to Airflow UI]({AIRFLOW_UI_BASE_URL})
    """)
    st.markdown("---")
    if st.button("🔄 Reset Application State", help="Clear all uploaded files, analysis results, and reset app state."):
        # Clear all files in uploads and processed_data
        for directory in [UPLOADS_DIR, PROCESSED_DATA_DIR]:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    app_logger.info(f"Deleted: {item_path}")
        
        st.session_state.clear()
        # Ensure default values are set after clearing for a clean restart
        st.session_state['documents'] = []
        st.session_state['selected_document_id'] = None
        st.success("App state and all temporary files reset!")
        time.sleep(1) # Give user time to see message
        st.rerun()

# --- Main Content Area ---
st.markdown("---")
st.header("1. Upload Your Document(s) 📄")

# Initialize session state variables
if 'documents' not in st.session_state:
    st.session_state['documents'] = []
if 'selected_document_id' not in st.session_state:
    st.session_state['selected_document_id'] = None

# Store the current list of uploaded files from the widget
current_uploaded_files = st.file_uploader(
    "Drag and drop your document(s) here or click to browse",
    type=["txt", "jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True, # Allow multiple files
    key="file_uploader"
)

# Process newly uploaded files only if the widget's value has changed
if current_uploaded_files:
    # Get names of files already in session state
    existing_file_names = {doc['name'] for doc in st.session_state['documents']}
    
    new_files_to_add = []
    for uploaded_file in current_uploaded_files:
        if uploaded_file.name not in existing_file_names:
            new_files_to_add.append(uploaded_file)
            
    if new_files_to_add:
        for uploaded_file in new_files_to_add:
            file_path_in_uploads = os.path.join(UPLOADS_DIR, uploaded_file.name)
            try:
                # Clear previous files in uploads to ensure only one is processed per run
                # This logic is now handled by the "Reset" button or when a new analysis is triggered
                # For multiple uploads, we want to keep them until explicitly cleared.
                with open(file_path_in_uploads, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                app_logger.info(f"File '{uploaded_file.name}' saved to {UPLOADS_DIR}")
                
                # Add document to session state
                doc_id = f"{uploaded_file.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state['documents'].append({
                    'id': doc_id,
                    'name': uploaded_file.name,
                    'path': file_path_in_uploads,
                    'file_prefix': get_file_prefix(uploaded_file.name),
                    'status': 'uploaded',
                    'dag_run_id': None,
                    'kpis': {},
                    'scores': {},
                    'last_update': datetime.now()
                })
            except Exception as e:
                st.error(f"❌ Error saving file '{uploaded_file.name}': {e}")
                app_logger.error(f"Error saving uploaded file: {e}", exc_info=True)
        st.rerun() # Rerun only if new files were added

# Display uploaded documents and their status
if st.session_state['documents']:
    st.subheader("Uploaded Documents:")
    doc_data = []
    for doc in st.session_state['documents']:
        doc_data.append({
            "Document Name": doc['name'],
            "Status": doc['status'].replace('_', ' ').title(),
            "Airflow Run ID": doc['dag_run_id'] if doc['dag_run_id'] else "N/A"
        })
    st.dataframe(pd.DataFrame(doc_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("2. Start Analysis for Documents 🚀")
    st.write("Click 'Start Analysis' for each document you wish to process.")

    # Iterate through documents to show analysis buttons/status
    for i, doc in enumerate(st.session_state['documents']):
        col_name, col_button, col_status = st.columns([3, 2, 3])
        with col_name:
            st.markdown(f"**{doc['name']}**")
        
        with col_button:
            if doc['status'] in ['uploaded', 'error', 'completed']: # Allow re-trigger if error or completed
                if st.button(f"🚀 Start Analysis", key=f"analyze_btn_{doc['id']}"):
                    with st.spinner(f"Initiating Airflow DAG for {doc['name']}..."):
                        # Clear processed data for this specific file before new run
                        for item in os.listdir(PROCESSED_DATA_DIR):
                            if item.startswith(doc['file_prefix']):
                                item_path = os.path.join(PROCESSED_DATA_DIR, item)
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                    app_logger.info(f"Deleted old processed file for {doc['name']}: {item_path}")

                        dag_trigger_response = trigger_airflow_dag(AIRFLOW_DAG_ID, doc['name'])
                        if dag_trigger_response and 'dag_run_id' in dag_trigger_response:
                            st.session_state['documents'][i]['dag_run_id'] = dag_trigger_response['dag_run_id']
                            st.session_state['documents'][i]['status'] = 'triggered'
                            st.session_state['documents'][i]['last_update'] = datetime.now()
                            st.success(f"✅ DAG triggered for {doc['name']}! Run ID: `{st.session_state['documents'][i]['dag_run_id']}`")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.session_state['documents'][i]['status'] = 'error'
                            st.error(f"❌ Failed to trigger DAG for {doc['name']}.")
            elif doc['status'] == 'triggered' or doc['status'] == 'processing':
                with col_status:
                    st.info(f"⏳ Processing...")
                    st.progress(0) # Placeholder, actual progress is polled below

    st.markdown("---")
    st.header("3. Processing Status & Results ✨")

    # Polling for documents that are currently processing
    processing_docs = [doc for doc in st.session_state['documents'] if doc['status'] in ['triggered', 'processing']]
    if processing_docs:
        st.subheader("Documents Currently Processing:")
        progress_placeholders = {}
        status_placeholders = {}
        for doc in processing_docs:
            st.markdown(f"**{doc['name']}**")
            status_placeholders[doc['id']] = st.empty()
            progress_placeholders[doc['id']] = st.progress(0)

        all_completed = False
        polling_retries = 0
        max_polling_retries = 180 # 30 minutes (180 * 10 seconds)

        while not all_completed and polling_retries < max_polling_retries:
            all_completed = True
            for i, doc in enumerate(st.session_state['documents']):
                if doc['status'] in ['triggered', 'processing']:
                    file_prefix = get_file_prefix(doc['name'])
                    if check_for_specific_output_files(file_prefix):
                        st.session_state['documents'][i]['status'] = 'completed'
                        st.session_state['documents'][i]['last_update'] = datetime.now()
                        app_logger.info(f"Output files found for {doc['name']}. Marking as completed.")
                        status_placeholders[doc['id']].empty()
                        progress_placeholders[doc['id']].empty()
                        st.success(f"🎉 Analysis for **{doc['name']}** complete!")
                        st.balloons()
                        # Set this as the selected document to view immediately
                        st.session_state['selected_document_id'] = doc['id']
                    else:
                        all_completed = False # At least one doc is still processing
                        current_progress = (polling_retries + 1) / max_polling_retries
                        status_placeholders[doc['id']].info(f"⏳ Airflow DAG run `{doc['dag_run_id']}` in progress. Checking for results... (Attempt {polling_retries+1}/{max_polling_retries})")
                        progress_placeholders[doc['id']].progress(current_progress)
            
            if not all_completed:
                time.sleep(10) # Wait for 10 seconds before checking again
                polling_retries += 1
        
        if not all_completed: # If loop finished due to timeout
            for i, doc in enumerate(st.session_state['documents']):
                if doc['status'] in ['triggered', 'processing']:
                    st.session_state['documents'][i]['status'] = 'error'
                    status_placeholders[doc['id']].empty()
                    progress_placeholders[doc['id']].empty()
                    st.error(f"❌ Analysis for **{doc['name']}** timed out or output files not found. Check [Airflow UI]({AIRFLOW_UI_BASE_URL}).")
                    app_logger.error(f"Analysis for {doc['name']} timed out.")
        st.rerun() # Rerun to update UI after polling loop

    # Display results for completed documents
    completed_docs = [doc for doc in st.session_state['documents'] if doc['status'] == 'completed']
    if completed_docs:
        st.subheader("View Analysis Results:")
        
        # Create a mapping for selectbox options
        doc_options = {doc['name']: doc['id'] for doc in completed_docs}
        
        # Determine initial selection
        current_selection_name = None
        if st.session_state['selected_document_id']:
            for doc in completed_docs:
                if doc['id'] == st.session_state['selected_document_id']:
                    current_selection_name = doc['name']
                    break
        
        # If no previous selection or selected doc is no longer completed, default to the first completed
        if not current_selection_name and completed_docs:
            current_selection_name = completed_docs[0]['name']
            st.session_state['selected_document_id'] = completed_docs[0]['id']

        selected_doc_name = st.selectbox(
            "Select a document to view its dashboard:",
            options=list(doc_options.keys()),
            index=list(doc_options.keys()).index(current_selection_name) if current_selection_name else 0,
            key="doc_selector"
        )
        
        # Update selected_document_id based on selectbox choice
        if selected_doc_name:
            st.session_state['selected_document_id'] = doc_options[selected_doc_name]
        
        # Find the selected document's data
        selected_doc_data = next((doc for doc in completed_docs if doc['id'] == st.session_state['selected_document_id']), None)

        if selected_doc_data:
            st.markdown(f"### Dashboard for: **{selected_doc_data['name']}**")
            
            # Load results if not already loaded (e.g., after rerun)
            if not selected_doc_data['kpis'] or not selected_doc_data['scores']:
                extracted_kpis, final_scores = load_output_data_for_document(selected_doc_data['file_prefix'])
                selected_doc_data['kpis'] = extracted_kpis
                selected_doc_data['scores'] = final_scores
                # Update session state with loaded data
                for i, doc in enumerate(st.session_state['documents']):
                    if doc['id'] == selected_doc_data['id']:
                        st.session_state['documents'][i] = selected_doc_data
                        break
            else:
                extracted_kpis = selected_doc_data['kpis']
                final_scores = selected_doc_data['scores']

            if extracted_kpis and final_scores:
                # --- Overall Score & Health Category ---
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    total_score = final_scores.get('total_score', 0.0)
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
                    st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
                    health_category = final_scores.get('health_category', {})
                    if health_category:
                        st.metric(label=f"Health Category {health_category.get('emoji', '')}", value=health_category.get('name', 'N/A'))
                        st.info(health_category.get('description', ''))
                    else:
                        st.warning("Health category could not be determined.")
                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    confidence_score = final_scores.get('confidence_score', 0.0)
                    prediction_method = final_scores.get('prediction_method', 'unknown')
                    
                    if confidence_score >= 80:
                        confidence_color = "#4CAF50"
                        confidence_label = "High"
                    elif confidence_score >= 60:
                        confidence_color = "#FFC107"
                        confidence_label = "Medium"
                    else:
                        confidence_color = "#F44336"
                        confidence_label = "Low"
                        
                    st.markdown(f"""
                        <div style="
                            background-color: #262730; 
                            padding: 15px; 
                            border-radius: 15px; 
                            text-align: center; 
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                            margin-bottom: 20px;
                        ">
                            <h3 style="color: #ADD8E6; margin-bottom: 10px;">Confidence</h3>
                            <p style="
                                font-size: 2.5em; 
                                font-weight: bold; 
                                color: {confidence_color}; 
                                line-height: 1;
                            ">{confidence_score:.1f}%</p>
                            <p style="color: white; font-size: 0.9em;">{confidence_label} ({prediction_method})</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                
                # --- Download Button ---
                results_json = json.dumps(final_scores, indent=2)
                st.download_button(
                    label="📥 Download Full Analysis Results (JSON)",
                    data=results_json,
                    file_name=f"{selected_doc_data['file_prefix']}_analysis_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                    mime="application/json",
                    help="Download the complete KPI and score data as a JSON file."
                )
                st.markdown("---")

                # --- Category Scores Chart & Table ---
                st.subheader("📊 Category Scores")
                category_scores_data = []
                for category, score in final_scores['category_scores'].items():
                    category_scores_data.append({"Category": category.replace('_', ' ').title(), "Score": score})
                
                if category_scores_data:
                    df_category_scores = pd.DataFrame(category_scores_data)
                    df_category_scores = df_category_scores.sort_values(by="Score", ascending=False) 

                    st.bar_chart(df_category_scores.set_index("Category"), use_container_width=True, height=300)
                    
                    with st.expander("View Raw Category Scores Table"):
                        df_category_scores['Score'] = df_category_scores['Score'].apply(lambda x: f"{x:.2f}")
                        st.dataframe(df_category_scores, use_container_width=True, hide_index=True)
                else:
                    st.info("No category scores available to display.")

                st.markdown("---")

                # --- Extracted KPIs ---
                with st.expander("📋 View Extracted KPIs", expanded=False):
                    kpi_data_for_df = [{"KPI": k, "Value": v} for k, v in extracted_kpis.items()]
                    st.dataframe(kpi_data_for_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # --- Normalized KPI Scores ---
                with st.expander("📈 View Normalized KPI Scores", expanded=False):
                    normalized_kpi_data_for_df = [{"KPI": k, "Normalized Score": f"{v:.2f}"} for k, v in final_scores['normalized_kpis'].items()]
                    st.dataframe(normalized_kpi_data_for_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")

                # --- Suggestions and Warnings ---
                st.subheader("💡 Insights & Recommendations")
                suggestions_and_warnings_list = final_scores.get('suggestions_and_warnings', [])
                
                if suggestions_and_warnings_list:
                    # Check if the last item is a dictionary (structured LLM output)
                    llm_structured_insights = None
                    if suggestions_and_warnings_list and isinstance(suggestions_and_warnings_list[-1], dict) and \
                       "overall_assessment" in suggestions_and_warnings_list[-1]:
                        llm_structured_insights = suggestions_and_warnings_list.pop(-1) # Get and remove from list
                    
                    # Display rule-based suggestions first
                    if suggestions_and_warnings_list:
                        st.markdown("#### Rule-Based Alerts:")
                        for item in suggestions_and_warnings_list:
                            st.markdown(f"- {item}")
                        st.markdown("---") # Separator for AI insights

                    # Display AI-Powered Insights
                    if llm_structured_insights:
                        st.markdown("#### AI-Powered Insights:")
                        
                        # Overall Assessment
                        if llm_structured_insights.get("overall_assessment"):
                            st.markdown(f"**Overall Assessment:** {llm_structured_insights['overall_assessment']}")

                        # Strengths by Category
                        if llm_structured_insights.get("strengths"):
                            st.markdown("\n**Strengths by Category:**")
                            for category, strength_list in llm_structured_insights["strengths"].items():
                                st.markdown(f"- **{category.replace('_', ' ').title()}:**")
                                if strength_list:
                                    for strength in strength_list:
                                        st.markdown(f"  - <span style='color: #4CAF50;'>✅</span> {strength}", unsafe_allow_html=True)
                                else:
                                    st.markdown("  - *No specific strengths identified for this category.*")
                            if not llm_structured_insights["strengths"]:
                                st.markdown("  - *No specific strengths identified by AI.*")

                        # Weaknesses by Category
                        if llm_structured_insights.get("weaknesses"):
                            st.markdown("\n**Weaknesses by Category:**")
                            for category, weakness_list in llm_structured_insights["weaknesses"].items():
                                st.markdown(f"- **{category.replace('_', ' ').title()}:**")
                                if weakness_list:
                                    for weakness in weakness_list:
                                        st.markdown(f"  - <span style='color: #FF6347;'>❌</span> {weakness}", unsafe_allow_html=True)
                                else:
                                    st.markdown("  - *No specific weaknesses identified for this category.*")
                            if not llm_structured_insights["weaknesses"]:
                                st.markdown("  - *No specific weaknesses identified by AI.*")

                        # Actionable Recommendations
                        if llm_structured_insights.get("recommendations"):
                            st.markdown("\n**Actionable Recommendations:**")
                            for rec in llm_structured_insights["recommendations"]:
                                st.markdown(f"- 💡 {rec}") # Using a consistent emoji for recommendations
                            if not llm_structured_insights["recommendations"]:
                                st.markdown("  - *No specific recommendations provided by AI.*")
                    else:
                        st.info("No AI-powered insights generated (e.g., LLM API key not configured or LLM error).")

                else:
                    st.info("No specific suggestions or warnings generated for this analysis.")

            else:
                st.error("❌ Could not load processed data for the selected document. Please check the 'processed_data' directory and Airflow logs.")
        else:
            st.info("Select a processed document from the dropdown above to view its dashboard.")
    else:
        st.info("No documents have been processed yet. Upload a document and click 'Start Analysis'.")

else:
    st.info("⬆️ Upload document(s) to begin the analysis.")