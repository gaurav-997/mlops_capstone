import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
MAX_LOG_SIZE = 5*1024*1024
BACKUP_COUNT = 2

# create log dir and files 
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir,LOG_DIR)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path = os.path.join(log_dir_path,LOG_FILE) 

def configure_logger():
    
    # get the logger 
    logger = logging.getLogger(name="data_logger")
    logger.setLevel(logging.DEBUG)

    # define console and file handler 
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # file_handler = logging.FileHandler(filename=log_file_path)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # define log format and put in handlers 
    formatter = logging.Formatter(fmt= '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add handlers to logger 
    logger.addHandler(console_handler)
    logger.addHandler(file_handler) 
    
configure_logger()
