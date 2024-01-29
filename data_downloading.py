import os
import logging
import gdown

from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime

FOLDER_1_URL = "https://drive.google.com/drive/folders/1UmMQ-eZjux6cpmm37OKH-AoSZ4W_RzeQ"
logging.basicConfig(level=logging.INFO, filename="./logs/downloading.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")



def downloading():

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


sched = BlockingScheduler(timezone = tzlocal.get_localzone_name())
sched.add_job(downloading, 'interval', minutes = 1)
sched.start()

