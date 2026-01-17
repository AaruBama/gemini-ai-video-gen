import logging
from datetime import datetime

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup and return a configured logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)