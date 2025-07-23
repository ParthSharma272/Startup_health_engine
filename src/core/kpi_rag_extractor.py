import json
import re
from typing import Dict, Any, Tuple, Optional, List
import openai
from src.utils.logger_config import logger

class KPIRAGExtractor:
    """
    Extracts structured KPI data from raw text using a Retrieval-Augmented Generation (RAG) approach
    with the OpenAI Large Language Model (LLM).
    This module is responsible for 'Step 2: KPI Extraction' from the pipeline.
    """

    def __init__(self, kpi_benchmarks: List[Dict[str, Any]], openai_api_key: str, kpi_benchmark_map: Dict[str, Any]):
        """
        Initializes the KPIRAGExtractor by loading the LLM and tokenizer.

        Args:
            kpi_benchmarks (List[Dict[str, Any]]): List of KPI benchmark dictionaries, used to
                                                   inform the LLM about expected KPIs.
            openai_api_key (str): Your OpenAI API key.
            kpi_benchmark_map (Dict[str, Any]): A map of KPI names to their benchmark configurations,
                                                 used to determine expected data types (e.g., 'predefined').
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = "gpt-3.5-turbo" # You can change this to "gpt-4o" for potentially better results
        self.kpi_list = [kpi['kpi'] for kpi in kpi_benchmarks]
        self.kpi_benchmark_map = kpi_benchmark_map # Store the map for type checking
        logger.info(f"KPIRAGExtractor initialized with OpenAI model: {self.model_name}.")

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
        example_json_output = '{"Monthly Recurring Revenue (MRR)": 100000, "Burn Rate": 50000, "Founder Commitment (Full-Time)": "Full-Time"}' # Corrected example KPI name

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
        Extracts KPIs from the given text content using the OpenAI LLM.

        Args:
            text_content (str): The raw text content from which to extract KPIs.

        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - Dict[str, Any]: A dictionary of extracted KPI names and their values.
                                 Returns an empty dict if no KPIs are extracted or parsing fails.
                - str: The raw text response received directly from the LLM.
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Cannot extract KPIs.")
            return {}, "OpenAI client failed to initialize."

        prompt = self._construct_prompt(text_content)
        logger.debug(f"Sending prompt to OpenAI LLM:\n{prompt}")

        llm_response_string = "" # Initialize to empty string
        parsed_kpis = {} # Initialize to empty dict

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured KPI data from business documents. Your output must be a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, # Instruct OpenAI to return JSON
                max_tokens=1024,
                temperature=0.0 # For extraction, we want deterministic output
            )

            if chat_completion.choices and chat_completion.choices[0].message and chat_completion.choices[0].message.content:
                llm_response_string = chat_completion.choices[0].message.content
            else:
                logger.warning(f"OpenAI API returned unexpected output: {chat_completion}")
                llm_response_string = "OpenAI API returned no generated text or unexpected format."

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

            # --- NEW DEBUG LOG HERE ---
            logger.debug(f"Parsed KPIs before cleaning (from LLM/fallback): {parsed_kpis}")
            # --- END NEW DEBUG LOG ---

            # Final cleaning and type conversion for parsed KPIs
            cleaned_kpis = {}
            for kpi_name, value in parsed_kpis.items():
                # Get KPI info from the map to determine expected type
                kpi_info = self.kpi_benchmark_map.get(kpi_name)
                
                if kpi_info and kpi_info.get('normalization') == 'predefined':
                    # For predefined KPIs, keep the value as a string, strip whitespace
                    # Also, ensure the string matches one of the predefined params keys if possible
                    cleaned_string_value = str(value).strip() if value is not None else None
                    if cleaned_string_value and kpi_info.get('params'):
                        # Try to find a case-insensitive match among predefined options
                        matched_predefined_value = next(
                            (k for k in kpi_info['params'] if k.lower() == cleaned_string_value.lower()),
                            cleaned_string_value # Fallback to original if no exact predefined match
                        )
                        cleaned_kpis[kpi_name] = matched_predefined_value
                    else:
                        cleaned_kpis[kpi_name] = cleaned_string_value
                    logger.debug(f"Predefined KPI '{kpi_name}' kept as string: '{cleaned_kpis[kpi_name]}'")
                elif isinstance(value, str):
                    # For other KPIs, attempt numeric conversion
                    # Remove currency symbols (₹, $) AND COMMAS, then convert to float
                    cleaned_value = re.sub(r'[₹$]', '', value).replace(',', '').strip()
                    try:
                        if cleaned_value.endswith('%'):
                            cleaned_kpis[kpi_name] = float(cleaned_value.replace('%', ''))
                        else:
                            cleaned_kpis[kpi_name] = float(cleaned_value)
                        logger.debug(f"Numeric KPI '{kpi_name}' converted to float: {cleaned_kpis[kpi_name]}")
                    except ValueError:
                        # If it's a string that can't be converted to float, keep as string
                        cleaned_kpis[kpi_name] = value
                        logger.warning(f"Could not convert '{value}' to float for KPI '{kpi_name}'. Keeping as string.")
                else:
                    # Keep non-string, non-predefined values as they are (e.g., already numbers from JSON)
                    cleaned_kpis[kpi_name] = value
                    logger.debug(f"KPI '{kpi_name}' kept as original type: {cleaned_kpis[kpi_name]}")

            return cleaned_kpis, llm_response_string

        except openai.APIError as e:
            logger.error(f"OpenAI API error during KPI extraction: {e}", exc_info=True)
            return {}, f"OpenAI API error: {e}"
        except Exception as e:
            logger.error(f"General error during OpenAI KPI extraction: {e}", exc_info=True)
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
        # Split by comma followed by a space, but only if the space is followed by an uppercase letter (start of a new KPI)
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
                    # No numeric conversion here; let the main extract_kpis loop handle type conversion
                    parsed_data[matched_kpi_name] = value_extracted
                else:
                    logger.warning(f"Extracted KPI '{kpi_name_extracted}' could not be mapped to an official KPI. Skipping.")
            else:
                logger.warning(f"Could not parse KPI entry: '{entry}'. Skipping.")

        return parsed_data
