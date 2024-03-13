import gc
import logging.config
from ..config.logging_config import dict_config
import warnings
from datetime import datetime

import dill
import pandas as pd
import tzlocal
from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.model_selection import KFold, cross_val_score

from .Tools import Tools

warnings.filterwarnings('ignore')


logging.config.dictConfig(dict_config)

train_logger = logging.getLogger('train_logger')
train_logger.setLevel(logging.INFO)

FOLDER_1_URL = "https://drive.google.com/drive/folders/1cY4sleAzmJM1JuPqt_bxAKWK-BDn0WaK"

tools = Tools()
RANDOM_SEED = 42


model = tools.model
model_metadata = tools.model_metadata
preprocessor = tools.preprocessor

def model_training():

    score = 0

    # try:
    #     data = gdown.download_folder(FOLDER_1_URL, quiet=True, use_cookies=False)
    #     for file in data:
    #         df_name = str(file)[str(file).rfind('/') + 1: len(str(file))]
    #         logging.info(f'File {df_name} has been downloaded')
    #
    #     files = os.listdir('24news_practice/practice/buff/')
    #     for file in files:
    #         file_size = os.path.getsize('./practice/' + file)
    #         if file_size > 0:
    #             logging.info(f'File {file} size ok')
    #         else:
    #             logging.error(f'File {file} size did\'t matchcp')
    # except Exception as e:
    #     logging.error(f'Download failed: {e}')


    try:
        # df_clickhouse = pd.read_csv('24news_practice/practice/internship/clickhouse.csv')
        # logging.info(f'Dataframe internship/clickhouse.csv read successfully')
        new_df_clickhouse = pd.read_csv('24news_practice/practice/buff/clickhouse.csv')
        train_logger.info(f'Dataframe buff/clickhouse.csv read successfully')

        # df_creatives = pd.read_csv('24news_practice/practice/internship/creatives.csv')
        # logging.info(f'Dataframe internship/creatives.csv read successfully')
        new_df_creatives = pd.read_csv('24news_practice/practice/buff/creatives.csv')
        train_logger.info(f'Dataframe buff/creatives.csv read successfully')

        try:
            new_df_creatives.drop(['ai_title_id', 'original_id', 'rating', 'rating_editor', 'rating_updated'], axis = 1, inplace = True)
            train_logger.info(f'df_creatives redundant parameters deleted successfully')

        except Exception as e:
            train_logger.error(f'df_creatives redundant parameters delete error: {e}')


        # # df_leads = pd.read_csv('24news_practice/practice/internship/leads.csv')
        # # logging.info(f'Dataframe internship/leads.csv read successfully')
        # new_df_leads = pd.read_csv('24news_practice/practice/buff/leads.csv')
        # logging.info(f'Dataframe buff/leads.csv read successfully')
    except Exception as e:
        train_logger.error(f'Dataframe read error: {e}')
       


    # df_creatives = df_creatives.rename(columns={'id': 'creative_id'})
    new_df_creatives = new_df_creatives.rename(columns={'id': 'creative_id'})

    # df_clickhouse = pd.concat([df_clickhouse, new_df_clickhouse], ignore_index=True)
    # try:
    #     df_clickhouse.to_csv('24news_practice/internship/clickhouse_last.csv', index = False)
    #     logging.info(f'Full dataframe internship/clickhouse.csv saved successfully')
    #
    #     df_creatives = pd.concat([df_creatives, new_df_creatives], ignore_index=True)
    #     df_creatives.to_csv('24news_practice/internship/creatives_last.csv', index=False)
    #     logging.info(f'Full dataframe internship/creatives.csv saved successfully')
    #
    #     df_leads = pd.concat([df_leads, new_df_leads], ignore_index=True)
    #     df_leads.to_csv('24news_practice/internship/leads_last.csv', index=False)
    #     logging.info(f'Full dataframe internship/leads.csv saved successfully')
    #
    # except Exception as e:
    #     logging.error(f'Dataframe save error: {e}')
    #     print(f'Dataframe save error: {e}')
    #
    # del df_leads
    # gc.collect()

    df = pd.merge(new_df_clickhouse, new_df_creatives.drop(['campaign_id', 'stream_id', 'iab_category'], axis = 1), on="creative_id")

    df = tools.paused_status_dropping(df)

    df = tools.nan_filling(tools.dub_dropping(tools.nan_click_viewed__deleting(df)))

    df = tools.place_number_decrease(df)

    df = tools.enter_utm_term_prep(df)

    del new_df_clickhouse
    del new_df_creatives
    gc.collect()

    X_train = df.drop('click', axis=1)
    y_train = df['click']

    del df
    gc.collect()

    print('preps')
    try:
        X_train_prep = preprocessor.transform(X_train).A
        train_logger.info(f'Train dataset preparation successfully')
    except Exception as e:
        print(e)
        train_logger.error(f'Train dataset preparation error: {e}')
        

    try:
        kf = KFold(n_splits = 3, shuffle=True, random_state = RANDOM_SEED)
        scores = cross_val_score(model, X_train_prep, y_train, cv=kf, scoring='accuracy', n_jobs=-1,
                                 error_score='raise')
        score = scores.mean()
        train_logger.info(f'Cross val score calculating successfully, roc_auc = {score}')
    except Exception as e:
        train_logger.error(f'Cross val score calculating error: {e}')
        score = model_metadata

    print('fit')
    try:
        hist = model.fit(X_train_prep, y_train, epochs=5, batch_size=1024, verbose=0)
        train_logger.info(f'model {type(model).__name__} fit successfully')
    except Exception as e:
        train_logger.error(f'Model fit error: {e}')
        


    fit_time = datetime.now()
    fit_time = fit_time.strftime('%Y_%m_%d_%H_%M')
    prep_tools_dict = {"model_metadata": f"Model: {type(model).__name__}, Accuracy: {max(hist.history['accuracy'])}, last_fit_time: {fit_time}"}
    prep_tools_dict['preprocessor'] = tools.preprocessor
    prep_tools_dict['req_df_columns'] = tools.req_df_columns

    
    try:
        with open(f'24news_practice/practice/pickles/prep_tools_dict_tf.pkl', 'wb') as f:
            dill.Pickler(f, recurse=True).dump(prep_tools_dict)
        train_logger.info(f'model_dict save successfully')

    except Exception as e:
        print(e)
        train_logger.error(f'model_dict save error: {e}')
        

    try:
        model_json = model.to_json()
        with open("24news_practice/practice/pickles/tf_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("24news_practice/practice/pickles/tf_weights.h5")
        train_logger.info(f'Saved fitted model to disk')
    except Exception as e:
        train_logger.error(f'model save error: {e}')
        


    del X_train
    del y_train
    del X_train_prep
    gc.collect()
    train_logger.info(f'Model train ended')


model_training()


def start_train():
    train_logger.info(f'Model train started')
    sched = BlockingScheduler(timezone = tzlocal.get_localzone_name())
    sched.add_job(model_training, 'interval', weeks = 1)
    sched.start()



