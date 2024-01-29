import gc

import dill
import gdown
import os
import logging
import pandas as pd
from .settings import settings
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from .Tools import Tools
from datetime import datetime

logging.basicConfig(level=logging.INFO, filename="../logs/Train.log",filemode="a", format="%(asctime)s %(levelname)s %(message)s")
FOLDER_1_URL = "https://drive.google.com/drive/folders/1cY4sleAzmJM1JuPqt_bxAKWK-BDn0WaK"

tools = Tools()
def model_training():

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

    try:

        with open(settings.MODEL_PICKLE_PATH, 'rb') as in_strm:
            model_dict = dill.load(in_strm)

        model = model_dict['model']
        model_metadata = model_dict['metadata']
        logging.info(f'Model {model_metadata} from {settings.MODEL_PICKLE_PATH} loaded for train')

    except Exception  as e:
        logging.error(f'Model {model_metadata} from {settings.MODEL_PICKLE_PATH} didn\'t loaded, error: {e}')

    try:

        with open(settings.PREPROCESSOR_PICKLE_PATH, 'rb') as in_strm:
            preprocessor_pipline_dict = dill.load(in_strm)

        preprocessor = preprocessor_pipline_dict['preprocessor']

        logging.info(f'Preprocessor loaded from  {settings.PREPROCESSOR_PICKLE_PATH} for model train')

    except Exception as e:
        logging.error(f'Preprocessor from {settings.MODEL_PICKLE_PATH} didn\'t loaded error: {e}')

    df_clickhouse = pd.read_csv('../internship/clickhouse.csv')
    new_df_clickhouse = pd.read_csv('../API/clickhouse.csv')

    df_creatives = pd.read_csv('../internship/creatives.csv')
    new_df_creatives = pd.read_csv('../API/creatives.csv')

    df_leads = pd.read_csv('../internship/leads.csv')
    new_df_leads = pd.read_csv('../API/leads.csv')

    df_creatives = df_creatives.rename(columns={'id': 'creative_id'})
    new_df_creatives = new_df_creatives.rename(columns={'id': 'creative_id'})

    df_clickhouse = pd.concat([df_clickhouse, new_df_clickhouse], ignore_index=True)
    df_clickhouse.to_csv('../internship/clickhouse_last.csv', index = False)

    df_creatives = pd.concat([df_creatives, new_df_creatives], ignore_index=True)
    df_creatives.to_csv('../internship/creatives_last.csv', index=False)

    df_leads = pd.concat([df_leads, new_df_leads], ignore_index=True)
    df_leads.to_csv('../internship/leads_last.csv', index=False)

    del df_leads
    gc.collect()

    df = pd.merge(df_clickhouse, df_creatives, on="creative_id")

    df = tools.nan_filling(tools.dub_dropping(tools.nan_click_viewed__deleting(df)))

    del df_clickhouse
    del df_creatives
    gc.collect()

    X_train = df.drop('click', axis=1)
    y_train = df['click']

    del df
    gc.collect()

    X_train_prep = preprocessor.fit_transform(X_train, y_train)

    model.model.fit(X_train_prep, y_train)

    model_dict = {}
    fit_time = datetime.now()
    fit_time = fit_time.strftime('%Y_%m_%d_%H_%M')
    model_dict['metadata'] = 'Модель: HistGradientBoostingClassifier, ROC_AUC test: 0.6728202352712497'
    model_dict['model'] = model
    with open(f'../pickles/model_dict_{fit_time}.pkl', 'wb') as f:
        dill.Pickler(f, recurse=True).dump(model_dict)








sched = BlockingScheduler(timezone = tzlocal.get_localzone_name())
sched.add_job(model_training(), 'interval', days = 1)
sched.start()

