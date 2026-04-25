import logging
import os
from logging.handlers import RotatingFileHandler
from core.config import config # <-- Import the config

def setup_logger(name="drone_ai"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Fetch log path from config
    log_file = config.get("log_file", "logs/drone_editor.log")
    
    if not logger.handlers:
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s | PID:%(process)d | %(levelname)s | %(name)s | %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

log = setup_logger()