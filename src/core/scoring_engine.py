from typing import Dict, Any, List
from src.core.kpi_normalizer import KPINormalizer
from src.core.config_loader import ConfigLoader
from src.utils.logger_config import logger

class ScoringEngine:
    """
    The main scoring engine for startup health.

    Orchestrates KPI normalization, input validation, category score calculation,
    and total health score aggregation.
    """

    def __init__(self, config_loader: ConfigLoader):
        """
        Initializes the ScoringEngine.

        Args:
            config_loader (ConfigLoader): An instance of ConfigLoader to load configurations.
        """
        self.config_loader = config_loader
        self.kpi_configs: Dict[str, Any] = {}
        self.kpi_normalizer: KPINormalizer
        self.mandatory_kpis: List[str] = []

        self._load_configurations()
        self._initialize_normalizer()
        self._identify_mandatory_kpis()
        logger.info("ScoringEngine initialized successfully.")

    def _load_configurations(self):
        """Loads KPI benchmarks and weights from configuration files."""
        try:
            # This method correctly loads from config_loader
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
        # This correctly initializes with the kpi_benchmark_map
        self.kpi_normalizer = KPINormalizer(self.kpi_benchmark_map)

    def _identify_mandatory_kpis(self):
        """Identifies all mandatory KPIs based on the weights configuration."""
        for category_kpis in self.kpi_weights['kpi_weights_within_category'].values():
            for kpi_name, weight_info in category_kpis.items():
                if weight_info.get('mandatory', False):
                    self.mandatory_kpis.append(kpi_name)
        logger.info(f"Identified mandatory KPIs: {self.mandatory_kpis}")

    def _validate_input_kpis(self, kpi_dict: Dict[str, Any]) -> List[str]:
        """
        Validates if all mandatory KPIs are present in the input dictionary.

        Args:
            kpi_dict (Dict[str, Any]): The input dictionary of raw KPI values.

        Returns:
            List[str]: A list of missing mandatory KPIs.
        """
        missing_kpis = [kpi for kpi in self.mandatory_kpis if kpi not in kpi_dict or kpi_dict[kpi] is None]
        if missing_kpis:
            logger.error(f"Missing mandatory KPIs: {', '.join(missing_kpis)}")
        return missing_kpis

    def calculate_scores(self, kpi_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the health score for a startup based on its KPIs.

        Args:
            kpi_dict (Dict[str, Any]): A dictionary of raw KPI values.

        Returns:
            Dict[str, Any]: A dictionary containing normalized KPI scores, category scores, and total score.
                            Returns an error structure if mandatory KPIs are missing.
        """
        missing_kpis = self._validate_input_kpis(kpi_dict)
        if missing_kpis:
            return {
                "error": "Missing mandatory KPIs",
                "missing_kpis": missing_kpis,
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
            "total_score": 0.0
        }

        # Step 1: Normalize all provided KPIs
        for kpi_display_name, raw_value in kpi_dict.items():
            # THIS IS THE CRITICAL LINE: Ensure 'kpi_dict' is passed as 'all_raw_kpis'
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

        # Step 2: Calculate Category Health Scores (weighted average)
        for category, data in category_weighted_sums.items():
            if data["weight_sum"] > 0:
                output_scores["category_scores"][category] = round(data["score_sum"] / data["weight_sum"], 2)
                logger.debug(f"Calculated category '{category}' score: {output_scores['category_scores'][category]:.2f}")
            else:
                output_scores["category_scores"][category] = 0.0 # No KPIs for this category or no weight assigned
                logger.debug(f"No valid KPIs or weights for category '{category}'. Score set to 0.0.")

        # Step 3: Aggregate Category Health Scores to get Total Score (weighted average)
        total_score_sum = 0.0
        total_category_weight_sum = 0.0
        for category, category_score in output_scores["category_scores"].items():
            # Corrected: Access the 'weight' key from the nested dictionary
            category_overall_weight = self.kpi_weights['category_weights'].get(category, {}).get("weight", 0.0)
            total_score_sum += category_score * category_overall_weight
            total_category_weight_sum += category_overall_weight
            logger.debug(f"Adding category '{category}' score ({category_score:.2f}) with weight {category_overall_weight} to total.")


        if total_category_weight_sum > 0:
            output_scores["total_score"] = round(total_score_sum / total_category_weight_sum, 2)
            logger.info(f"Calculated total score: {output_scores['total_score']:.2f}")
        else:
            output_scores["total_score"] = 0.0
            logger.warning("No valid category weights found. Total score set to 0.0.")

        return output_scores
