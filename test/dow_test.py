import gdown
import os
import logging

logging.basicConfig(level=logging.INFO, filename="Train.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")

FOLDER_1_URL = "https://drive.google.com/drive/folders/1Lm1GCvwt0QFC9KKbfdap0fbLyHq0d5zh"

try:
    data = gdown.download_folder(FOLDER_1_URL, quiet=True, use_cookies=False)
    for file in data:
        df_name = str(file)[str(file).rfind('/') + 1: len(str(file))]
        logging.info(f'File {df_name} has been downloaded')

    files = os.listdir('./practice/')
    for file in files:
        file_size = os.path.getsize('./practice/' + file)
        if file_size > 0:
            logging.info(f'File {file} size ok')
        else:
            logging.error(f'File {file} size did\'t matchcp')
except Exception as e:
    logging.error(f'Download failed: {e}')
