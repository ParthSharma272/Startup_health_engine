import math
import re
from typing import Dict, Any, Union, List
from src.utils.logger_config import logger

class KPINormalizer:
    def __init__(self, kpi_benchmark_map: Dict[str, Dict[str, Any]]):
        self.kpi_benchmark_map = kpi_benchmark_map
        logger.info("KPINormalizer initialized with KPI benchmark map.")

    def _evaluate_formula(self, formula_str: str, current_kpi_numeric_value: Union[float, int], params: Dict[str, Any], kpi_name_for_logging: str) -> float:
        """
        Evaluates a mathematical formula string using provided KPI value and parameters.
        """
        # Define a safe execution environment for eval.
        eval_context = {
            'math': math,
            'min': min,
            'max': max,
            'log': math.log,
            'value': current_kpi_numeric_value, # The KPI value itself is named 'value' in formulas
        }

        # Add parameters from the benchmark config directly to eval_context
        for param_key, param_val in params.items():
            # Ensure param_val is not None if it's expected to be used in arithmetic
            if param_val is None:
                logger.warning(f"Parameter '{param_key}' for KPI '{kpi_name_for_logging}' is None. Setting to 0 for formula evaluation.")
                eval_context[param_key] = 0 # Default to 0 if a parameter is None
            else:
                eval_context[param_key] = param_val

        try:
            # Evaluate the formula using the custom context (which acts as locals)
            result = eval(formula_str, {"__builtins__": None}, eval_context)
            return float(result)
        except ZeroDivisionError:
            logger.error(f"ZeroDivisionError in formula '{formula_str}' for KPI '{kpi_name_for_logging}' with value '{current_kpi_numeric_value}' and params '{params}'.", exc_info=True)
            raise ValueError(f"ZeroDivisionError in formula for KPI '{kpi_name_for_logging}'.")
        except (SyntaxError, NameError, TypeError, ValueError) as e:
            logger.error(f"Error evaluating formula '{formula_str}' for KPI '{kpi_name_for_logging}' with value '{current_kpi_numeric_value}' and params '{params}': {e}. Eval context keys: {list(eval_context.keys())}", exc_info=True)
            raise ValueError(f"Malformed formula or invalid input for KPI '{kpi_name_for_logging}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during formula evaluation for KPI '{kpi_name_for_logging}': {e}", exc_info=True)
            raise

    def normalize_kpi(self, kpi_name: str, kpi_value: Any, all_raw_kpis: Dict[str, Any]) -> float:
        """
        Normalizes a single KPI value based on its defined benchmark formula.
        Returns a score between 0 and 100.
        """
        # Handle None values at the very beginning
        if kpi_value is None:
            logger.warning(f"KPI '{kpi_name}' has a None value. Returning 0.0.")
            return 0.0

        benchmark_info = self.kpi_benchmark_map.get(kpi_name)
        if not benchmark_info:
            logger.warning(f"No benchmark found for KPI '{kpi_name}'. Returning 0.0.")
            return 0.0

        normalization_method = benchmark_info.get('normalization')
        formula_str = benchmark_info.get('formula')
        params = benchmark_info.get('params', {})

        try:
            # Handle predefined scores for categorical KPIs
            if normalization_method == "predefined":
                if isinstance(kpi_value, str):
                    mapping = params # Use the params dict directly as the mapping
                    # Case-insensitive match for predefined values
                    for key, score in mapping.items():
                        if kpi_value.lower() == key.lower():
                            return float(score)
                    logger.warning(f"Predefined KPI '{kpi_name}' value '{kpi_value}' not found in mapping. Returning 0.0.")
                    return 0.0
                else:
                    logger.warning(f"Predefined KPI '{kpi_name}' has non-string value '{kpi_value}'. Returning 0.0.")
                    return 0.0
            
            # For numerical KPIs: Ensure value is numeric before passing to formula evaluator
            numeric_value: Union[float, int]
            if isinstance(kpi_value, str):
                # KPIRAGExtractor should ideally convert these, but as a safeguard
                cleaned_value_str = re.sub(r'[₹$]', '', kpi_value).replace(',', '').strip()
                try:
                    if cleaned_value_str.endswith('%'):
                        numeric_value = float(cleaned_value_str.replace('%', ''))
                    elif re.match(r'^\d+\s*:\s*\d+\.?\d*$', cleaned_value_str): # Handle ratios like "1:9.1"
                        parts = re.split(r'\s*:\s*', cleaned_value_str)
                        if float(parts[1]) != 0:
                            numeric_value = float(parts[0]) / float(parts[1])
                        else:
                            numeric_value = 0.0
                            logger.warning(f"Ratio KPI '{kpi_name}' has zero in denominator: '{cleaned_value_str}'. Setting to 0.")
                    # Handle time units (e.g., "18 Hours", "15.5 months")
                    elif re.match(r'^\d+\.?\d*\s*(months|month|hours|hour|days|day)$', cleaned_value_str, re.IGNORECASE): # Handle time units
                        num_part = re.search(r'^\d+\.?\d*', cleaned_value_str).group(0)
                        numeric_value = float(num_part)
                    elif re.match(r'^\d+\.?\d*\s*(ftes?|fte)$', cleaned_value_str, re.IGNORECASE): # Handle FTE units
                        num_part = re.search(r'^\d+\.?\d*', cleaned_value_str).group(0)
                        numeric_value = float(num_part)
                    else:
                        numeric_value = float(cleaned_value_str)
                except ValueError:
                    logger.warning(f"KPI '{kpi_name}' has unparseable numeric string value '{kpi_value}'. Returning 0.0.")
                    return 0.0
            else:
                try:
                    numeric_value = float(kpi_value)
                except (ValueError, TypeError):
                    logger.warning(f"KPI '{kpi_name}' has non-numeric value '{kpi_value}' for numeric normalization. Returning 0.0.")
                    return 0.0

            # For numerical KPIs requiring formula evaluation
            if formula_str:
                normalized_score = self._evaluate_formula(
                    formula_str,
                    numeric_value, # Pass the pre-processed numeric value
                    params,
                    kpi_name
                )
                return max(0.0, min(100.0, normalized_score)) # Ensure score is within 0-100
            else:
                logger.warning(f"Numeric KPI '{kpi_name}' has no formula defined. Returning 0.0.")
                return 0.0
        except Exception as e:
            logger.error(f"Failed to normalize KPI '{kpi_name}' with value '{kpi_value}': {e}", exc_info=True)
            return 0.0

