import json
import os
from typing import Dict, Any, List
from src.utils.logger_config import logger

class ConfigLoader:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.kpi_weights_path = os.path.join(config_dir, 'kpi_weights.json')
        self.kpi_benchmarks_path = os.path.join(config_dir, 'kpi_benchmarks.json')
        self.percentile_thresholds_path = os.path.join(config_dir, 'percentile_thresholds.json')
        logger.info(f"ConfigLoader initialized with config directory: {config_dir}")
        
        # Verify all config files exist
        for config_file in [self.kpi_weights_path, self.kpi_benchmarks_path, self.percentile_thresholds_path]:
            if not os.path.exists(config_file):
                logger.error(f"Required config file not found: {config_file}")
            else:
                logger.info(f"Config file found: {config_file}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads a JSON configuration file.
        """
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_path}: {e}")
            raise ValueError(f"Invalid JSON in {config_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {config_path}: {e}", exc_info=True)
            raise

    def get_kpi_benchmark_map(self, kpi_benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converts a list of KPI benchmark configurations into a dictionary
        mapping KPI names to their full configurations for easier lookup.
        """
        kpi_map = {kpi['kpi']: kpi for kpi in kpi_benchmarks}
        logger.debug("KPI benchmark map created.")
        return kpi_map

