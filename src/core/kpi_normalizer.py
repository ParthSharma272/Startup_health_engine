import math
import re
from typing import Dict, Any, Union, List
from src.utils.logger_config import logger

class KPINormalizer:

    def __init__(self, kpi_benchmark_map: Dict[str, Dict[str, Any]]):

        self.kpi_benchmark_map = kpi_benchmark_map
        self.kpi_formula_aliases = self._create_kpi_aliases()
        logger.info("KPINormalizer initialized with KPI formula aliases.")

    def _create_kpi_aliases(self) -> Dict[str, str]:

        alias_map = {}
        for full_name in self.kpi_benchmark_map.keys():
            # Try to extract alias from parentheses (e.g., MRR from Monthly Recurring Revenue (MRR))
            match = re.search(r'\((.*?)\)', full_name)
            if match:
                alias = match.group(1).strip()
            else:

                # Remove common suffixes and spaces to create a clean alias
                alias = full_name.replace(' (%)', '').replace('%', '').replace(' ', '').replace('(inMonths)', '').strip()
            

            # Replace hyphens with underscores, remove special characters
            alias = re.sub(r'[^a-zA-Z0-9_]', '', alias)
            if not alias: # Fallback if cleaning results in empty string
                alias = full_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '').replace('-', '_').strip()

            alias_map[full_name] = alias
        logger.debug(f"KPI Aliases created: {alias_map}")
        return alias_map

    def _evaluate_formula(self, formula_str: str, current_kpi_numeric_value: Union[float, int], params: Dict[str, Any], kpi_name_for_logging: str) -> float:

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
            raise
        except (SyntaxError, NameError, TypeError, ValueError) as e:
            logger.error(f"Error evaluating formula '{formula_str}' for KPI '{kpi_name_for_logging}' with value '{current_kpi_numeric_value}' and params '{params}': {e}. Eval context keys: {list(eval_context.keys())}", exc_info=True)
            raise ValueError(f"Malformed formula or invalid input: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during formula evaluation for KPI '{kpi_name_for_logging}': {e}", exc_info=True)
            raise

    def normalize_kpi(self, kpi_name: str, kpi_value: Any, all_raw_kpis: Dict[str, Any]) -> float:

        # Handle None values at the very beginning
        if kpi_value is None:
            logger.warning(f"KPI '{kpi_name}' has a None value. Returning 0.0.")
            return 0.0

        benchmark_info = self.kpi_benchmark_map.get(kpi_name)
        if not benchmark_info:
            logger.warning(f"No benchmark found for KPI '{kpi_name}'. Returning 0.0.")
            return 0.0

        normalization_method = benchmark_info['normalization']
        formula_str = benchmark_info.get('formula') # Use .get to handle 'predefined' without formula
        params = benchmark_info.get('params', {})

        try:
            # Handle predefined scores for categorical KPIs
            if normalization_method == "predefined":
                if isinstance(kpi_value, str):
                    mapping = params # Use the params dict directly as the mapping
                    return float(mapping.get(kpi_value, 0.0))
                else:
                    logger.warning(f"Predefined KPI '{kpi_name}' has non-string value '{kpi_value}'. Returning 0.0.")
                    return 0.0

            # For numerical KPIs: Preprocess value before passing to formula evaluator
            numeric_value: Union[float, int]
            if isinstance(kpi_value, str):
                # Remove currency symbols (₹, $) AND COMMAS, then convert to float
                cleaned_value_str = re.sub(r'[₹$]', '', kpi_value).replace(',', '').strip()
                try:
                    if cleaned_value_str.endswith('%'):
                        numeric_value = float(cleaned_value_str.replace('%', ''))
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

