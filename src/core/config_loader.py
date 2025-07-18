import json
import os
from typing import Dict, Any
from src.utils.logger_config import logger

class ConfigLoader:
    """
    Handles loading and validating configuration files for the scoring engine.
    """

    def __init__(self, config_dir: str = 'config'):
        """
        Initializes the ConfigLoader.

        Args:
            config_dir (str): The directory where configuration JSON files are located.
        """
        self.config_dir = config_dir
        self.kpi_benchmarks_path = os.path.join(self.config_dir, 'kpi_benchmarks.json')
        self.kpi_weights_path = os.path.join(self.config_dir, 'kpi_weights.json')

    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Loads a JSON configuration file.

        Args:
            file_path (str): The full path to the JSON configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration as a dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            Exception: For other unexpected errors during file loading.
        """
        if not os.path.exists(file_path):
            logger.error(f"Configuration file not found: {file_path}")
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            logger.info(f"Successfully loaded configuration from {file_path}")
            return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
            raise

    def load_all_configs(self) -> Dict[str, Any]:
        """
        Loads all necessary configuration files (benchmarks and weights).

        Returns:
            Dict[str, Any]: A dictionary containing 'kpi_benchmarks' and 'kpi_weights'.

        Raises:
            Exception: If any configuration file fails to load.
        """
        all_configs = {}
        all_configs['kpi_benchmarks'] = self.load_config(self.kpi_benchmarks_path)
        all_configs['kpi_weights'] = self.load_config(self.kpi_weights_path)
        return all_configs

    def get_kpi_benchmark_map(self, kpi_benchmarks: list) -> Dict[str, Dict[str, Any]]:
        """
        Creates a dictionary for quick lookup of benchmark data by KPI name.

        Args:
            kpi_benchmarks (list): List of KPI benchmark dictionaries.

        Returns:
            Dict[str, Dict[str, Any]]: A map from KPI name to its benchmark details.
        """
        return {item['kpi']: item for item in kpi_benchmarks}
