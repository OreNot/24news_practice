import logging
from tensorflow.keras.models import model_from_json
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import time

import dill
import numpy as np
import pandas as pd

from .Creatives import Creatives
from .settings import settings

import warnings
warnings.filterwarnings('ignore')


def timeit(func):
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print(f'Processing time of {func.__qualname__}(): {int((time.time() - start_time) * 1000)} ms.')
        # % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


logging.basicConfig(level=logging.INFO, filename="practice/logs/Tools.log", filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")

df_creatives = Creatives().df_creatives


class Tools:

    def __init__(self):
        with open(settings.PREP_TOOLS_DICT_PATH, 'rb') as in_strm:
            prep_tools_dict = dill.load(in_strm)

        
        self.req_df_columns = prep_tools_dict['req_df_columns']  # .drop('click')
        self.preprocessor = prep_tools_dict['preprocessor']
        
        self.model_metadata = prep_tools_dict['model_metadata']
        # logging.info(f'Encoder loaded from  {settings.PREP_TOOLS_DICT_PATH}')
        self.cretive_tag_df = pd.DataFrame(columns=['imp_id', 'tag_id', 'plcmtcnt', 'creatives_list_id', 'creative_id'])
        self.req_df = pd.DataFrame(columns=self.req_df_columns)

        json_file = open(settings.KERAS_MODEL_JSON, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(settings.KERAS_MODEL_WIGHTS)
        logging.info(f'Loaded model from disk  {settings.KERAS_MODEL_JSON}')

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])

    #@timeit
    def nan_filling(self, X, y=None):  # Функция заполнения пропущенных значений
        nan_cols = {}
        for col in X.columns:
            nan_count = X[col].isna().sum()
            if nan_count != 0:
                per = np.round((nan_count / X.shape[0]) * 100, 2)
                nan_cols[col] = per

        for col, per in nan_cols.items():
            if per < 5:
                X = X[X[col].isna() == False]
            else:
                X[col] = X[col].fillna('unknown')

        return X

    #@timeit
    def with_creatives_mergding(self, X):  # Функция мержа с датафреймом креативов

        X = pd.merge(X, df_creatives, on="creative_id")
        return X

    #@timeit
    def get_creatives_imps_df(self, ssp_req,
                              req_datetime_str):  # Функция получения датафрейма imps - creative_id и датафрейма запроса из запроса

        creatives_dict = dict()

        for key, value in ssp_req.creatives_list.items():
            creatives_dict[key] = set().union(*(d.keys() for d in value))

        cretive_tag_list = []
        req_list = []

        for imp in ssp_req.imps:
            for creative in creatives_dict[imp['creatives_list_id']]:
                # with parallel_backend('threading', n_jobs = -1):
                cretive_tag_list.append({
                    'imp_id': imp['id'],
                    'tag_id': imp['tagid'],
                    'plcmtcnt': imp['plcmtcnt'],
                    'creative_id': creative,
                    'creatives_list_id': imp['creatives_list_id'],
                })

                req_list.append({
                    'site_id': ssp_req.site_id,
                    'OS': ssp_req.os,
                    'browser': ssp_req.browser,
                    'device': ssp_req.device,
                    'geo_country': ssp_req.country,
                    'geo_city': ssp_req.city,
                    'site_id': ssp_req.site_id,
                    'loss_reason': ssp_req.news_category,
                    'enter_utm_source': ssp_req.us,
                    'enter_utm_medium': ssp_req.um,
                    'enter_utm_content': ssp_req.uct,
                    'enter_utm_term': ssp_req.ut,
                    'creative_id': creative,
                    'event_date_time': req_datetime_str,
                    'tag_id': imp['tagid'],
                    'place_number': imp['seq']
                })

        # with parallel_backend('threading', n_jobs = -1):
        cretive_tag_df = pd.DataFrame(cretive_tag_list,
                                      columns=['imp_id', 'tag_id', 'plcmtcnt', 'creatives_list_id', 'creative_id'])
        req_df = pd.DataFrame(req_list, columns=self.req_df_columns)

        req_df = req_df.fillna(0)
        return cretive_tag_df, req_df

   # @timeit
    def get_result_dict(self, imps_list, X):  # Функция формирования результирующего словаря

        result_dict = dict()

        for imp in imps_list:
            result_dict[imp] = [{row['creative_id']: row['CPM']} for index, row in
                                (X.where(X['imp_id'] == imp).dropna()[['creative_id', 'CPM']].reset_index()).iterrows()]
        return result_dict

    #@timeit
    def nan_click_viewed__deleting(self, X, y=None):  # Функция удаления строк с пустыми значениями клика и с view == 0
        X = X[X['click'].isna() == False]
        X = X[X['view'] != 0]
        return X

    #@timeit
    def dub_dropping(self, X, y=None):  # Функция удаления дубликатов примеров
        X = X.drop_duplicates()
        return X

    #@timeit
    def paused_status_dropping(self, X, y=None):
        X.drop(X.loc[X['status'] == 'paused'].index, inplace=True)
        return X

    #@timeit
    def place_number_decrease(self, X, y=None):
        X = X[X['place_number'] > 0]
        X['place_number'] = X['place_number'] - 1
        return X

    def enter_utm_term_prep(self, X, y=None):
        X['enter_utm_term'] = X['enter_utm_term'].apply(lambda x: str(x).split('_')[0])
        return X