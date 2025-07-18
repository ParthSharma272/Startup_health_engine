import json
import re
from typing import Dict, Any, Tuple, Optional, List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.logger_config import logger

class KPIRAGExtractor:
    """
    Extracts structured KPI data from raw text using a Retrieval-Augmented Generation (RAG) approach
    with a local Large Language Model (LLM).
    This module is responsible for 'Step 2: KPI Extraction' from the pipeline.
    """

    def __init__(self, kpi_benchmarks: List[Dict[str, Any]]):
        """
        Initializes the KPIRAGExtractor by loading the LLM and tokenizer.

        Args:
            kpi_benchmarks (List[Dict[str, Any]]): List of KPI benchmark dictionaries, used to
                                                   inform the LLM about expected KPIs.
        """
        self.model_name = "google/flan-t5-large" # Using large model
        self.tokenizer = None
        self.model = None
        self._load_llm()
        self.kpi_list = [kpi['kpi'] for kpi in kpi_benchmarks]
        logger.info("KPIRAGExtractor initialized.")

    def _load_llm(self):
        """
        Loads the pre-trained Hugging Face LLM and tokenizer.
        Ensures the model is loaded to CPU for broader compatibility.
        """
        try:
            if self.tokenizer is None or self.model is None:
                logger.info(f"Loading Hugging Face model: {self.model_name} to CPU.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.device = "cpu"
                logger.info(f"Hugging Face model loaded successfully on device: {self.device}.")
        except Exception as e:
            logger.critical(f"Failed to load Hugging Face model {self.model_name}: {e}", exc_info=True)
            self.tokenizer = None
            self.model = None
            raise

    def _construct_prompt(self, text_content: str) -> str:
        """
        Constructs the prompt for the LLM to extract KPIs.

        Args:
            text_content (str): The raw text extracted from the document.

        Returns:
            str: The formatted prompt for the LLM.
        """
        kpi_names_str = ", ".join(self.kpi_list)
        
        # Adding a few-shot example to guide the LLM more effectively
        example_document = "Our MRR is $100,000. Burn Rate is $50,000. Founder commitment is Full-Time."
        example_json_output = '{"Monthly Recurring Revenue (MRR)": 100000, "Burn Rate": 50000, "Founder Commitment": "Full-Time"}'

        prompt = f"""
        Analyze the following business document and extract the specified Key Performance Indicators (KPIs) and their numerical or string values.
        Your response MUST be a valid JSON object.
        For each KPI found, provide its value. If a KPI is not explicitly mentioned, omit it from the JSON.
        Ensure numerical values are extracted as numbers (integers or floats), and categorical values as strings.

        KPIs to extract: {kpi_names_str}

        Example:
        Document:
        ---
        {example_document}
        ---
        JSON Output:
        {example_json_output}

        Document:
        ---
        {text_content}
        ---

        JSON Output:
        """
        return prompt.strip()

    def extract_kpis(self, text_content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extracts KPIs from the given text content using the loaded LLM.

        Args:
            text_content (str): The raw text content from which to extract KPIs.

        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - Dict[str, Any]: A dictionary of extracted KPI names and their values.
                                 Returns an empty dict if no KPIs are extracted or parsing fails.
                - str: The raw text response received directly from the LLM.
        """
        if not self.model or not self.tokenizer:
            logger.error("LLM model or tokenizer not loaded. Cannot extract KPIs.")
            return {}, "LLM model or tokenizer failed to load during initialization."

        prompt = self._construct_prompt(text_content)
        logger.debug(f"Sending prompt to LLM:\n{prompt}")

        llm_response_string = "" # Initialize to empty string
        parsed_kpis = {} # Initialize to empty dict

        try:
            text_generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1, # Explicitly use CPU
                max_new_tokens=1024, # Increased max_new_tokens to allow for longer JSON output
                min_length=10,
                do_sample=False, # For extraction, we generally want deterministic output
            )
            
            llm_output = text_generator(prompt)
            if llm_output and isinstance(llm_output, list) and len(llm_output) > 0 and 'generated_text' in llm_output[0]:
                llm_response_string = llm_output[0]['generated_text']
            else:
                logger.warning(f"LLM text_generator returned unexpected output: {llm_output}")
                llm_response_string = "LLM returned no generated text or unexpected format."

            logger.debug(f"Raw LLM response received:\n{llm_response_string}")
            
            # --- Robust JSON/KPI Parsing Logic ---
            json_match = re.search(r"\{.*\}", llm_response_string, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                try:
                    parsed_kpis = json.loads(json_string)
                    logger.info("Successfully parsed KPIs from LLM response as JSON.")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from LLM response: {e}. Attempting fallback parsing. Raw JSON string: {json_string[:500]}...", exc_info=True)
                    parsed_kpis = self._parse_comma_separated_kpis(llm_response_string)
            else:
                logger.warning("No JSON object found in LLM response. Attempting fallback parsing (comma-separated).")
                parsed_kpis = self._parse_comma_separated_kpis(llm_response_string)

            # Final cleaning and type conversion for parsed KPIs
            cleaned_kpis = {}
            for kpi_name, value in parsed_kpis.items():
                if isinstance(value, str):
                    # Remove currency symbols (₹, $) AND COMMAS, then convert to float
                    cleaned_value = re.sub(r'[₹$]', '', value).replace(',', '').strip()
                    try:
                        if cleaned_value.endswith('%'):
                            cleaned_kpis[kpi_name] = float(cleaned_value.replace('%', ''))
                        else:
                            cleaned_kpis[kpi_name] = float(cleaned_value)
                    except ValueError:
                        # If it's a string that can't be converted to float, keep as string
                        cleaned_kpis[kpi_name] = value
                else:
                    cleaned_kpis[kpi_name] = value
            
            return cleaned_kpis, llm_response_string

        except Exception as e:
            logger.error(f"Error during LLM KPI extraction: {e}", exc_info=True)
            return {}, llm_response_string if llm_response_string else f"Error during LLM inference: {e}"

    def _parse_comma_separated_kpis(self, text: str) -> Dict[str, Any]:
        """
        Parses a string of comma-separated 'Key: Value' pairs into a dictionary using a more robust splitting strategy.
        This is a fallback if the LLM doesn't produce valid JSON.
        """
        parsed_data = {}
        
        # Normalize the input string: remove leading/trailing spaces, and ensure consistent ": " delimiter
        text = text.strip()
        text = re.sub(r'\s*:\s*', ': ', text) # Replace any amount of whitespace around colon with ': '
        kpi_entries = re.split(r',\s*(?=[A-Z])', text)
        
        logger.debug(f"Raw text for parsing in _parse_comma_separated_kpis: {text}")
        logger.debug(f"Split KPI entries: {kpi_entries}")

        for entry in kpi_entries:
            parts = entry.split(': ', 1) 
            if len(parts) == 2:
                kpi_name_extracted = parts[0].strip()
                value_extracted = parts[1].strip()
                
                logger.debug(f"Extracted in _parse_comma_separated_kpis: KPI Name='{kpi_name_extracted}', Value='{value_extracted}'")

                # Normalize the extracted KPI name for matching against self.kpi_list
                kpi_name_normalized = kpi_name_extracted.strip(' "')

                matched_kpi_name = None
                # Prioritize exact match
                if kpi_name_normalized in self.kpi_list:
                    matched_kpi_name = kpi_name_normalized
                else:
                    # Attempt substring matching (case-insensitive) for robustness
                    for official_kpi_name in self.kpi_list:
                        if kpi_name_normalized.lower() in official_kpi_name.lower() or \
                           official_kpi_name.lower() in kpi_name_normalized.lower():
                            matched_kpi_name = official_kpi_name
                            break
                
                if matched_kpi_name:
                    converted_value: Any = value_extracted # Default to string
                    
                    # Further clean value_extracted to remove units like "months" before numeric conversion
                    if isinstance(value_extracted, str):
                        # Remove common unit suffixes (e.g., "months", "%") at the end of the string
                        value_without_units = re.sub(r'\s*(?:months|month|%)\s*$', '', value_extracted, flags=re.IGNORECASE).strip()
                        
                        # Now, remove currency symbols and ALL commas for numeric conversion
                        cleaned_numeric_value_str = re.sub(r'[₹$]', '', value_without_units).replace(',', '').strip()
                        
                        logger.debug(f"Cleaned numeric value string for '{matched_kpi_name}': '{cleaned_numeric_value_str}' (from original '{value_extracted}')")
                        
                        try:
                            converted_value = float(cleaned_numeric_value_str)
                            logger.debug(f"Converted '{matched_kpi_name}' to float: {converted_value}")
                        except ValueError:
                            logger.warning(f"Could not convert '{value_extracted}' to float for KPI '{matched_kpi_name}'. Keeping as string.")
                            converted_value = value_extracted # Keep original if conversion fails
                
                    parsed_data[matched_kpi_name] = converted_value
                else:
                    logger.warning(f"Extracted KPI '{kpi_name_extracted}' could not be mapped to an official KPI. Skipping.")
            else:
                logger.warning(f"Could not parse KPI entry: '{entry}'. Skipping.")

        return parsed_data
