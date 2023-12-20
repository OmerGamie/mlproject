import logging
import os
from datetime import datetime

#Create logs folder if not exists
LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
logs_path=os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

#Create log file path
LOG_FILE_PATH=os.path.join(os.getcwd(), logs_path, LOG_FILE)

#Create logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    
    )


    