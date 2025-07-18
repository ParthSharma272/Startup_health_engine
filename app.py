import streamlit as st
import os
import sys
import json
import io
import re
from datetime import datetime
import logging
import torch 

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your core modules
from src.core.config_loader import ConfigLoader
from src.core.kpi_extractor import KPIExtractor
from src.core.kpi_rag_extractor import KPIRAGExtractor
from src.core.scoring_engine import ScoringEngine
from src.utils.logger_config import logger, setup_logging

st.set_page_config(layout="wide", page_title="Startup Health Score Dashboard")
setup_logging(log_level='INFO') # Default to INFO initially

# Define paths relative to the app.py script
BASE_DIR = project_root
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# Ensure necessary directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Initialize ConfigLoader (cached to run once)
@st.cache_resource
def get_config_loader_instance():
    """Caches the ConfigLoader instance, pointing to the config directory."""
    return ConfigLoader(config_dir=CONFIG_DIR)

config_loader = get_config_loader_instance()

# Load KPI benchmarks and weights using ConfigLoader (cached)
@st.cache_resource
def load_all_configs_from_files(_config_loader):
    """Caches all configurations loaded from files using ConfigLoader."""
    try:
        # Ensure the config files exist before attempting to load
        if not os.path.exists(os.path.join(CONFIG_DIR, 'kpi_benchmarks.json')):
            raise FileNotFoundError(f"Missing config file: {os.path.join(CONFIG_DIR, 'kpi_benchmarks.json')}")
        if not os.path.exists(os.path.join(CONFIG_DIR, 'kpi_weights.json')):
            raise FileNotFoundError(f"Missing config file: {os.path.join(CONFIG_DIR, 'kpi_weights.json')}")
            
        return _config_loader.load_all_configs()
    except Exception as e:
        st.error(f"Error loading configurations from files: {e}. Please ensure 'config' directory and JSON files exist and are valid.")
        logger.critical(f"Failed to load configurations at app startup: {e}")
        st.stop() # Stop the app if essential configs cannot be loaded

all_configs = load_all_configs_from_files(config_loader)
kpi_benchmarks = all_configs['kpi_benchmarks']
kpi_benchmark_map = config_loader.get_kpi_benchmark_map(kpi_benchmarks)
kpi_weights = all_configs['kpi_weights']

# Initialize KPIExtractor (cached)
@st.cache_resource
def get_kpi_extractor():
    """Caches the KPIExtractor instance."""
    return KPIExtractor()

kpi_extractor = get_kpi_extractor()

# Initialize KPIRAGExtractor (cached - model loading is heavy!)
@st.cache_resource
def get_kpi_rag_extractor(_kpi_benchmarks):
    """Caches the KPIRAGExtractor instance and its loaded LLM."""
    return KPIRAGExtractor(_kpi_benchmarks)

kpi_rag_extractor = get_kpi_rag_extractor(kpi_benchmarks)

# Initialize ScoringEngine (cached)
@st.cache_resource
def get_scoring_engine(_config_loader):
    """
    Caches the ScoringEngine instance, passing the config_loader
    so it can load its own configs internally.
    """
    return ScoringEngine(config_loader=_config_loader)

scoring_engine = get_scoring_engine(config_loader)


st.title("📈 Startup Health Score Dashboard (Local Prototype)")
st.markdown("""
Upload a business document (TXT, JPG, PNG, PDF) to extract key performance indicators (KPIs),
review and edit them, and then calculate a comprehensive health score for the startup.
All processing is done locally within this application.
""")

# Sidebar for controls and info
with st.sidebar:
    st.header("Settings")
    log_level_select = st.selectbox(
        "Logging Level",
        ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
        index=0, # Default to INFO
        key='log_level_select'
    )
    st.session_state['log_level'] = log_level_select
    setup_logging(getattr(logging, log_level_select))

    st.markdown("---")
    st.info("Ensure Tesseract OCR is installed on your system for image/PDF processing.")
    st.info("This app uses a local Hugging Face model (`google/flan-t5-large`) for KPI extraction.")
    st.markdown("---")
    if st.button("Reset Application"):
        st.session_state.clear()
        st.rerun()

# --- File Upload Section ---
st.header("1. Upload Document")
uploaded_file = st.file_uploader(
    "Choose a document file (TXT, JPG, PNG, PDF)",
    type=["txt", "jpg", "jpeg", "png", "pdf"],
    key="file_uploader"
)

if uploaded_file is not None:
    if 'current_doc_id' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state['current_doc_id'] = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(uploaded_file.name).replace('.', '_')}"
        st.session_state['uploaded_file_name'] = uploaded_file.name
        st.session_state['raw_text'] = None # Clear previous raw text
        st.session_state['llm_raw_response'] = None # Clear previous LLM raw response
        st.session_state['extracted_kpis'] = None # Clear previous extracted KPIs
        st.session_state['edited_kpis'] = None # Clear previous edited KPIs
        st.session_state['final_scores_display'] = None # Clear previous final scores
        st.session_state['kpi_extraction_error'] = None # Clear previous KPI extraction error

    unique_doc_id = st.session_state['current_doc_id']
    original_filename = st.session_state['uploaded_file_name']
    
    # --- Step 1: Extract Raw Text ---
    st.header("2. Extracted Raw Text")
    if st.session_state.get('raw_text') is None:
        with st.spinner(f"Extracting text from {original_filename}... This might take a moment for images/PDFs."):
            # Save the uploaded file temporarily to read it
            temp_file_path = os.path.join(UPLOADS_DIR, original_filename)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"Uploaded file saved to: {temp_file_path}")

            raw_text = kpi_extractor.extract_text_from_document(temp_file_path)
            if raw_text:
                st.session_state['raw_text'] = raw_text
                st.success("Text extraction complete!")
            else:
                st.error(f"Failed to extract text from {original_filename}. Please check the file format and content.")
                st.session_state['raw_text'] = "" # Set to empty string to prevent re-running this step
                
            # Rerun to proceed to next step after text extraction
            st.rerun()

    # Display raw text immediately after extraction (or if already in session state)
    if st.session_state.get('raw_text'):
        with st.expander("View Raw Extracted Text"):
            st.text_area("Raw Text", st.session_state['raw_text'], height=300, disabled=True)
    elif st.session_state.get('raw_text') == "": # If extraction failed and raw_text is empty string
        st.warning("Raw text extraction resulted in an empty document. KPI extraction will be skipped.")


    # --- Step 2: Extract KPIs using RAG (LLM) ---

    if st.session_state.get('raw_text') and st.session_state.get('extracted_kpis') is None:
        st.header("3. Extracting KPIs with AI")
        with st.spinner("Extracting KPIs using LLM... This will take a moment (especially the first time)."):
            try:
                parsed_kpis, raw_llm_response = kpi_rag_extractor.extract_kpis(st.session_state['raw_text'])
                
                st.session_state['extracted_kpis'] = parsed_kpis
                st.session_state['llm_raw_response'] = raw_llm_response
                st.session_state['edited_kpis'] = parsed_kpis.copy()
                st.session_state['kpi_extraction_error'] = None
                
                if parsed_kpis:
                    st.success("KPI extraction complete!")
                else:
                    st.warning("KPI extraction completed but returned an empty set of KPIs. Check document content and LLM output below.")

            except Exception as e:
                st.error(f"An error occurred during KPI extraction: {e}")
                logger.error(f"Error during KPI extraction: {e}", exc_info=True)
                st.session_state['extracted_kpis'] = {}
                st.session_state['edited_kpis'] = {}
                st.session_state['llm_raw_response'] = "Error occurred, raw response not available."
                st.session_state['kpi_extraction_error'] = str(e)
            
            st.rerun()
    
    # --- Debugging: Display LLM Raw Response and Parsed KPIs ---
    if st.session_state.get('extracted_kpis') is not None or st.session_state.get('kpi_extraction_error'):
        st.markdown("---")
        st.subheader("Debugging: LLM Output & Parsed KPIs")
        with st.expander("View LLM Raw Response and Parsed KPIs"):
            if st.session_state.get('llm_raw_response'):
                st.write("**LLM Raw Response:**")
                st.text_area("LLM Raw Output", st.session_state['llm_raw_response'], height=200, disabled=True)
            else:
                st.info("LLM raw response not available (e.g., due to an early error).")
            
            st.write("**Parsed KPIs (before editing):**")
            if st.session_state.get('extracted_kpis'):
                st.json(st.session_state['extracted_kpis'])
            else:
                st.warning("No KPIs were successfully parsed from the LLM output.")
            
            if st.session_state.get('kpi_extraction_error'):
                st.error(f"KPI Extraction Error: {st.session_state['kpi_extraction_error']}")


    # --- Step 3: Display Editable KPIs and Calculate Scores ---
    if st.session_state.get('extracted_kpis') is not None:
        st.header("4. Review & Calculate Health Score")
        
        if st.session_state['extracted_kpis']:
            st.markdown("Review and adjust the extracted KPI values below:")
            
            with st.form(key=f"kpi_edit_form_{unique_doc_id}"):
                edited_kpis_current_run = {}
                col1, col2 = st.columns(2)
                kpi_items = list(st.session_state['extracted_kpis'].items())
                
                half = len(kpi_items) // 2 + (len(kpi_items) % 2)
                
                with col1:
                    for i in range(half):
                        kpi_name, kpi_value = kpi_items[i]
                        kpi_info = kpi_benchmark_map.get(kpi_name, {})
                        expected_type = 'string' if kpi_info.get('normalization') == 'predefined' else 'number'
                        
                        current_value_for_input = st.session_state['edited_kpis'].get(kpi_name, kpi_value)

                        if expected_type == 'number':
                            try:
                                current_value_for_input = float(current_value_for_input) if current_value_for_input is not None else 0.0
                            except (ValueError, TypeError):
                                current_value_for_input = 0.0
                            edited_kpis_current_run[kpi_name] = st.number_input(
                                f"{kpi_name}",
                                value=current_value_for_input,
                                key=f"kpi_input_{kpi_name}_col1_{unique_doc_id}",
                                format="%.2f"
                            )
                        else:
                            edited_kpis_current_run[kpi_name] = st.text_input(
                                f"{kpi_name}",
                                value=str(current_value_for_input) if current_value_for_input is not None else "",
                                key=f"kpi_input_{kpi_name}_col1_{unique_doc_id}"
                            )
                
                with col2:
                    for i in range(half, len(kpi_items)):
                        kpi_name, kpi_value = kpi_items[i]
                        kpi_info = kpi_benchmark_map.get(kpi_name, {})
                        expected_type = 'string' if kpi_info.get('normalization') == 'predefined' else 'number'

                        current_value_for_input = st.session_state['edited_kpis'].get(kpi_name, kpi_value)

                        if expected_type == 'number':
                            try:
                                current_value_for_input = float(current_value_for_input) if current_value_for_input is not None else 0.0
                            except (ValueError, TypeError):
                                current_value_for_input = 0.0
                            edited_kpis_current_run[kpi_name] = st.number_input(
                                f"{kpi_name}",
                                value=current_value_for_input,
                                key=f"kpi_input_{kpi_name}_col2_{unique_doc_id}",
                                format="%.2f"
                            )
                        else:
                            edited_kpis_current_run[kpi_name] = st.text_input(
                                f"{kpi_name}",
                                value=str(current_value_for_input) if current_value_for_input is not None else "",
                                key=f"kpi_input_{kpi_name}_col2_{unique_doc_id}"
                            )
                
                st.session_state['edited_kpis'] = edited_kpis_current_run
                
                submit_button = st.form_submit_button(label="Calculate Scores")

            if submit_button:
                with st.spinner("Calculating scores..."):
                    # This is the line that calls calculate_scores
                    final_scores = scoring_engine.calculate_scores(st.session_state['edited_kpis'])
                    st.session_state['final_scores_display'] = final_scores
                    st.success("Scores calculated!")
                st.rerun()
            
            if st.session_state.get('final_scores_display'):
                scores_to_display = st.session_state['final_scores_display']

                if "error" in scores_to_display and scores_to_display["error"]:
                    st.error(f"Scoring Error: {scores_to_display['error']}")
                    if "missing_kpis" in scores_to_display:
                        st.warning(f"Missing Mandatory KPIs: {', '.join(scores_to_display['missing_kpis'])}")
                else:
                    st.subheader(f"Total Health Score: {scores_to_display['total_score']:.2f} / 100")
                    
                    st.markdown("---")
                    st.subheader("Category Scores")
                    category_cols = st.columns(len(scores_to_display['category_scores']))
                    for i, (category, score) in enumerate(scores_to_display['category_scores'].items()):
                        with category_cols[i]:
                            st.metric(label=category.replace('_', ' ').title(), value=f"{score:.2f}")

                    st.markdown("---")
                    st.subheader("Normalized KPIs")
                    normalized_kpi_cols = st.columns(2)
                    normalized_kpi_items = list(scores_to_display['normalized_kpis'].items())
                    half_norm = len(normalized_kpi_items) // 2 + (len(normalized_kpi_items) % 2)

                    with normalized_kpi_cols[0]:
                        for i in range(half_norm):
                            kpi_name, score = normalized_kpi_items[i]
                            st.write(f"**{kpi_name}**: {score:.2f}")
                
                if st.button("Re-Calculate Scores (after edits)", key="recalculate_after_edits_button"):
                    with st.spinner("Re-calculating scores with your latest edits..."):
                        recalculated_scores = scoring_engine.calculate_scores(st.session_state['edited_kpis'])
                        st.session_state['final_scores_display'] = recalculated_scores
                        st.success("Scores re-calculated!")
                    st.rerun()

        else:
            st.info("No KPIs extracted. Please ensure your document contains relevant information.")
    elif st.session_state.get('kpi_extraction_error'):
        st.warning("Cannot proceed to KPI editing/scoring due to previous KPI extraction error.")
else:
    st.info("Upload a document to begin the analysis.")

