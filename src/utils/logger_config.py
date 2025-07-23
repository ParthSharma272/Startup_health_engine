import logging
import os

def setup_logging(log_level=logging.INFO):

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
