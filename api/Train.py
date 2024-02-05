import gc
import logging
import warnings
from datetime import datetime

import dill
import pandas as pd
import tzlocal
from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.model_selection import KFold, cross_val_score

from Tools import Tools

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/Train.log", filemode="a",
                        format="%(asctime)s %(levelname)s %(message)s")
FOLDER_1_URL = "https://drive.google.com/drive/folders/1cY4sleAzmJM1JuPqt_bxAKWK-BDn0WaK"

tools = Tools()
RANDOM_SEED = 42


model = tools.model
meta_data = tools
encoder = tools.encoder
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
        df_clickhouse = pd.read_csv('24news_practice/internship/clickhouse.csv')
        logging.info(f'Dataframe internship/clickhouse.csv read successfully')
        new_df_clickhouse = pd.read_csv('24news_practice/buff/clickhouse.csv')
        logging.info(f'Dataframe buff/clickhouse.csv read successfully')

        df_creatives = pd.read_csv('24news_practice/internship/creatives.csv')
        logging.info(f'Dataframe internship/creatives.csv read successfully')
        new_df_creatives = pd.read_csv('24news_practice/buff/creatives.csv')
        logging.info(f'Dataframe buff/creatives.csv read successfully')

        try:
            new_df_creatives.drop(['ai_title_id', 'original_id', 'rating', 'rating_editor', 'rating_updated'], axis = 1, inplace = True)
        except Exception as e:
            print(e)

        df_leads = pd.read_csv('24news_practice/internship/leads.csv')
        logging.info(f'Dataframe internship/leads.csv read successfully')
        new_df_leads = pd.read_csv('24news_practice/buff/leads.csv')
        logging.info(f'Dataframe buff/leads.csv read successfully')
    except Exception as e:
        logging.error(f'Dataframe read error: {e}')
        print(f'Dataframe read error: {e}')


    df_creatives = df_creatives.rename(columns={'id': 'creative_id'})
    new_df_creatives = new_df_creatives.rename(columns={'id': 'creative_id'})

    df_clickhouse = pd.concat([df_clickhouse, new_df_clickhouse], ignore_index=True)
    try:
        df_clickhouse.to_csv('24news_practice/internship/clickhouse_last.csv', index = False)
        logging.info(f'Full dataframe internship/clickhouse.csv saved successfully')

        df_creatives = pd.concat([df_creatives, new_df_creatives], ignore_index=True)
        df_creatives.to_csv('24news_practice/internship/creatives_last.csv', index=False)
        logging.info(f'Full dataframe internship/creatives.csv saved successfully')

        df_leads = pd.concat([df_leads, new_df_leads], ignore_index=True)
        df_leads.to_csv('24news_practice/internship/leads_last.csv', index=False)
        logging.info(f'Full dataframe internship/leads.csv saved successfully')

    except Exception as e:
        logging.error(f'Dataframe save error: {e}')
        print(f'Dataframe save error: {e}')

    del df_leads
    gc.collect()

    df = pd.merge(df_clickhouse, df_creatives, on="creative_id")

    df = tools.paused_status_dropping(df)

    df = tools.nan_filling(tools.dub_dropping(tools.nan_click_viewed__deleting(df)))

    df = tools.place_number_decrease(df)

    del df_clickhouse
    del df_creatives
    gc.collect()

    X_train = df.drop('click', axis=1)
    y_train = df['click']

    del df
    gc.collect()

    print('preps')
    try:
        X_train_prep = preprocessor.transform(X_train)
        logging.info(f'Train dataset preparation successfully')
    except Exception as e:
        logging.error(f'Train dataset preparation error: {e}')
        print(f'Train dataset preparation error: {e}')

    try:
        kf = KFold(n_splits = 3, shuffle=True, random_state = RANDOM_SEED)
        scores = cross_val_score(model, X_train_prep.toarray(), y_train, cv=kf, scoring='roc_auc', n_jobs=-1,
                                 error_score='raise')
        score = scores.mean()
        logging.info(f'Cross val score calculating successfully, roc_auc = {score}')
    except Exception as e:
        logging.error(f'Cross val score calculating error: {e}')
        score = model.model_metadata

    print('fit')
    try:
        model.fit(X_train_prep.toarray(), y_train)
        logging.info(f'model {type(model).__name__} fit successfully')
    except Exception as e:
        logging.error(f'Model fit error: {e}')
        print(f'Model fit error: {e}')


    fit_time = datetime.now()
    fit_time = fit_time.strftime('%Y_%m_%d_%H_%M')
    prep_tools_dict = {'model_metadata': f'Модель: {type(model).__name__}, ROC_AUC: {score}, last_fit_time: {fit_time}'}
    prep_tools_dict['model'] = model
    prep_tools_dict['encoder'] = tools.encoder
    prep_tools_dict['preprocessor'] = tools.preprocessor
    prep_tools_dict['req_df_columns'] = tools.req_df_columns

    print('saving')
    try:
        with open(f'24news_practice/pickles/prep_tools_dict.pkl', 'wb') as f:
            dill.Pickler(f, recurse=True).dump(prep_tools_dict)
        logging.info(f'model_dict save successfully')

    except Exception as e:
        logging.error(f'model_dict save error: {e}')
        print(f'model_dict save error: {e}')

    del X_train
    del y_train
    del X_train_prep
    gc.collect()


model_training()


sched = BlockingScheduler(timezone = tzlocal.get_localzone_name())
sched.add_job(model_training, 'interval', weeks = 1)
sched.start()

