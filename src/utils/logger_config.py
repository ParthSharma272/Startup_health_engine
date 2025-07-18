import logging
import os

def setup_logging(log_level=logging.INFO):
    """
    Sets up a standardized logging configuration for the application.

    Args:
        log_level (int): The minimum level of messages to log (e.g., logging.INFO, logging.DEBUG).
    """
    # Create a logger
    logger = logging.getLogger('startup_health_score')
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if already configured
    if not logger.handlers:
        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        logger.addHandler(ch)

    return logger

# Initialize logger when module is imported
logger = setup_logging()
