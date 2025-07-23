from typing import Dict, Any, List, Tuple
from src.core.kpi_normalizer import KPINormalizer
from src.core.config_loader import ConfigLoader
from src.utils.logger_config import logger

class ScoringEngine:

    def __init__(self, config_loader: ConfigLoader):

        self.config_loader = config_loader
        self.kpi_configs: Dict[str, Any] = {}
        self.kpi_normalizer: KPINormalizer
        self.mandatory_kpis: List[str] = []
        self.all_kpi_names: List[str] = [] # To store all expected KPI names

        self._load_configurations()
        self._initialize_normalizer()
        self._identify_mandatory_kpis()
        self._collect_all_kpi_names() # New method to collect all KPI names
        logger.info("ScoringEngine initialized successfully.")

    def _load_configurations(self):
        """Loads KPI benchmarks and weights from configuration files."""
        try:
            self.kpi_configs = self.config_loader.load_all_configs()
            self.kpi_benchmark_map = self.config_loader.get_kpi_benchmark_map(
                self.kpi_configs['kpi_benchmarks']
            )
            self.kpi_weights = self.kpi_configs['kpi_weights']
        except Exception as e:
            logger.critical(f"Failed to load scoring engine configurations: {e}")
            raise

    def _initialize_normalizer(self):
        """Initializes the KPI normalizer with loaded benchmarks."""
        self.kpi_normalizer = KPINormalizer(self.kpi_benchmark_map)

    def _identify_mandatory_kpis(self):
        """Identifies all mandatory KPIs based on the weights configuration."""
        self.mandatory_kpis = [] # Ensure it's reset
        for category_kpis in self.kpi_weights['kpi_weights_within_category'].values():
            for kpi_name, weight_info in category_kpis.items():
                if weight_info.get('mandatory', False):
                    self.mandatory_kpis.append(kpi_name)
        logger.info(f"Identified mandatory KPIs: {self.mandatory_kpis}")

    def _collect_all_kpi_names(self):
        """Collects all expected KPI names from the configuration."""
        self.all_kpi_names = [] # Ensure it's reset
        for category_kpis in self.kpi_weights['kpi_weights_within_category'].values():
            for kpi_name in category_kpis.keys():
                self.all_kpi_names.append(kpi_name)
        logger.info(f"Collected all expected KPI names: {self.all_kpi_names}")

    def _validate_input_kpis(self, kpi_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:

        missing_mandatory_kpis = [kpi for kpi in self.mandatory_kpis if kpi not in kpi_dict or kpi_dict[kpi] is None]

        missing_non_mandatory_kpis = []
        for kpi_name in self.all_kpi_names:
            if kpi_name not in self.mandatory_kpis and (kpi_name not in kpi_dict or kpi_dict[kpi_name] is None):
                missing_non_mandatory_kpis.append(kpi_name)

        if missing_mandatory_kpis:
            logger.error(f"Missing mandatory KPIs: {', '.join(missing_mandatory_kpis)}")
        if missing_non_mandatory_kpis:
            logger.warning(f"Missing non-mandatory KPIs: {', '.join(missing_non_mandatory_kpis)}")

        return missing_mandatory_kpis, missing_non_mandatory_kpis

    def calculate_scores(self, kpi_dict: Dict[str, Any]) -> Dict[str, Any]:

        missing_mandatory_kpis, missing_non_mandatory_kpis = self._validate_input_kpis(kpi_dict)
        if missing_mandatory_kpis:
            return {
                "error": "Missing mandatory KPIs",
                "missing_mandatory_kpis": missing_mandatory_kpis,
                "missing_non_mandatory_kpis": missing_non_mandatory_kpis, # Include for transparency
                "total_score": 0.0,
                "category_scores": {},
                "normalized_kpis": {}
            }

        normalized_kpis: Dict[str, float] = {}
        category_weighted_sums: Dict[str, Dict[str, float]] = {
            category: {"score_sum": 0.0, "weight_sum": 0.0}
            for category in self.kpi_weights['category_weights'].keys()
        }
        output_scores: Dict[str, Any] = {
            "normalized_kpis": {},
            "category_scores": {},
            "total_score": 0.0,
            "missing_non_mandatory_kpis": missing_non_mandatory_kpis # Add to output
        }

        # Step 1: Normalize all provided KPIs
        for kpi_display_name, raw_value in kpi_dict.items():
            # Only normalize KPIs that are expected in the config
            if kpi_display_name in self.all_kpi_names:
                normalized_score = self.kpi_normalizer.normalize_kpi(kpi_display_name, raw_value, kpi_dict)

                normalized_kpis[kpi_display_name] = normalized_score
                output_scores["normalized_kpis"][kpi_display_name] = round(normalized_score, 2)
                logger.debug(f"Normalized '{kpi_display_name}' (raw: {raw_value}) to {normalized_score:.2f}")

                # Map KPI to its category and apply its weight
                for category, kpis_in_category in self.kpi_weights['kpi_weights_within_category'].items():
                    if kpi_display_name in kpis_in_category:
                        weight_info = kpis_in_category[kpi_display_name]
                        kpi_weight_within_category = weight_info['weight']

                        # Add to category score using weighted sum
                        category_weighted_sums[category]["score_sum"] += normalized_score * kpi_weight_within_category
                        category_weighted_sums[category]["weight_sum"] += kpi_weight_within_category
                        break # KPI found in category, move to next KPI
            else:
                logger.warning(f"Input KPI '{kpi_display_name}' is not defined in KPI configurations. Skipping normalization and scoring.")


        # Step 2: Calculate Category Health Scores (weighted average)
        contributing_categories = []
        for category, data in category_weighted_sums.items():
            if data["weight_sum"] > 0:
                category_score = round(data["score_sum"] / data["weight_sum"], 2)
                output_scores["category_scores"][category] = category_score
                contributing_categories.append(category)
                logger.debug(f"Calculated category '{category}' score: {category_score:.2f}")
            else:
                output_scores["category_scores"][category] = 0.0 # No KPIs for this category or no weight assigned
                logger.debug(f"No valid KPIs or weights for category '{category}'. Score set to 0.0.")

        # Step 3: Aggregate Category Health Scores to get Total Score (weighted average)
        total_score_sum = 0.0
        total_category_overall_weight_sum = 0.0 # Renamed for clarity

        for category in contributing_categories: # Only iterate over categories that actually contributed
            category_score = output_scores["category_scores"][category]
            category_overall_weight = self.kpi_weights['category_weights'].get(category, {}).get("weight", 0.0)

            if category_overall_weight > 0: # Only add if the category has an overall weight
                total_score_sum += category_score * category_overall_weight
                total_category_overall_weight_sum += category_overall_weight
                logger.debug(f"Adding category '{category}' score ({category_score:.2f}) with overall weight {category_overall_weight} to total.")
            else:
                logger.warning(f"Category '{category}' has score but no overall weight defined. Skipping from total score calculation.")


        if total_category_overall_weight_sum > 0:
            output_scores["total_score"] = round(total_score_sum / total_category_overall_weight_sum, 2)
            logger.info(f"Calculated total score: {output_scores['total_score']:.2f}")
        else:
            output_scores["total_score"] = 0.0
            logger.warning("No valid category weights or contributing categories found. Total score set to 0.0.")

        return output_scores
