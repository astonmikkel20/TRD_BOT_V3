# TRD_BOT_V3/src/utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_path="logs/bot.log"):
    """
    Configura un logger global que:
     - Escribe en consola (nivel INFO y superior).
     - Escribe en archivo rotativo (5 MB por archivo, hasta 3 backups).
    """
    # Asegurarnos de que la carpeta exista
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Handler de archivo rotativo
    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    file_handler.setFormatter(logging.Formatter(file_fmt))
    logger.addHandler(file_handler)

    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(file_fmt))
    logger.addHandler(console_handler)
