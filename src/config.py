import json
from datetime import datetime
 
def fetch_config():
    with open ('../config.json','r') as file:
        config= json.load(file)
    return config
def log_error(error):
    config=fetch_config()
    file_path=config["log"]["log_file"]
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    modified_text = f"{timestamp} {str(error)}\n"

    with open(file_path, 'a') as file:
        file.write(modified_text)
    

    