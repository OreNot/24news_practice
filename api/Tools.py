from typing import List

import pandas as pd
import numpy as np
from .settings import settings
from .Encoder import Encoder
from .Creatives import Creatives
import logging
import time

def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time
logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/Tools.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")

df_creatives = object

df_creatives = Creatives().df_creatives

# Загрузка списка параметров объединенного датасета
req_df_columns = Encoder(settings.ENCODER_PICKLE_PATH).req_df_columns
class Tools:

    @timeit
    def nan_filling(self, X, y=None): # Функция заполнения пропущенных значений
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

    @timeit
    def with_creatives_mergding(self, X): # Функция мержа с датафреймом креативов

        X = pd.merge(X, df_creatives, on="creative_id")
        return X

    @timeit
    def get_creatives_imps_df(self, ssp_req, req_datetime_str): # Функция получения датафрейма imps - creative_id и датафрейма запроса из запроса

        X = pd.DataFrame(columns=['imp_id', 'tagid', 'plcmtcnt', 'creatives_list_id', 'creative_id'])
        y = pd.DataFrame(columns = req_df_columns)

        creatives_dict = dict()
        for key, value in ssp_req.creatives_list.items():
            creatives_dict[key] = set().union(*(d.keys() for d in value))

        print(creatives_dict)

        for imp in ssp_req.imps:
            #if imp['creatives_list_id'] == key:

                for creative in creatives_dict[key]:
                    new_row_X = {'imp_id': imp['id'],
                                 'tagid': imp['tagid'],
                                 'plcmtcnt': imp['plcmtcnt'],
                                 'creative_id': creative,
                                 'creatives_list_id': imp['creatives_list_id'],
                                 }
                    X.loc[len(X)] = new_row_X

                    new_row_y = {'site_id': ssp_req.site_id,
                                 'OS': ssp_req.os,
                                 'browser': ssp_req.browser,
                                 'device': ssp_req.device,
                                 'geo_country': ssp_req.country,
                                 'geo_city': ssp_req.city,
                                 'loss_reason': ssp_req.news_category,
                                 'enter_utm_source': ssp_req.us,
                                 'enter_utm_campaign': ssp_req.ucm,
                                 'enter_utm_medium': ssp_req.um,
                                 'enter_utm_content': ssp_req.uct,
                                 'enter_utm_term': ssp_req.ut,
                                 'creative_id': creative,
                                 'event_date_time': req_datetime_str

                                 }
                    y.loc[len(y)] = new_row_y


        # for key, value in ssp_req.creatives_list.items():
        #     for imp in ssp_req.imps:
        #         if imp['creatives_list_id'] == key:
        #             imp_id = imp['id']
        #             tagid = imp['tagid']
        #             plcmtcnt = imp['plcmtcnt']
        #             for list in value:
        #                 for key_, value_ in list.items():
        #                     new_row_X = {'imp_id': imp_id,
        #                                'tagid': tagid,
        #                                'plcmtcnt': plcmtcnt,
        #                                'creative_id': key_,
        #                                'creatives_list_id': key,
        #                                }
        #                     X.loc[len(X)] = new_row_X
        #                     new_row_y = {'site_id': ssp_req.site_id,
        #                                'OS': ssp_req.os,
        #                                'browser': ssp_req.browser,
        #                                'device': ssp_req.device,
        #                                'geo_country': ssp_req.country,
        #                                'geo_city': ssp_req.city,
        #                                'loss_reason': ssp_req.news_category,
        #                                'enter_utm_source': ssp_req.us,
        #                                'enter_utm_campaign': ssp_req.ucm,
        #                                'enter_utm_medium': ssp_req.um,
        #                                'enter_utm_content': ssp_req.uct,
        #                                'enter_utm_term': ssp_req.ut,
        #                                'creative_id': key_,
        #                                'event_date_time': req_datetime_str
        #
        #                                }
        #                     y.loc[len(y)] = new_row_y


        y = y.fillna(0)
        return X, y

    @timeit
    def get_result_dict(self, imps_list, X): # Функция формирования результирующего словаря

        result_dict = dict()

        for imp in imps_list:
            r = X.where(X['imp_id'] == imp).dropna()[['creative_id', 'CPM']]
            r = r.reset_index()
            ls = []
            for index, row in r.iterrows():
                dct = dict()
                dct[row['creative_id']] = row['CPM']
                ls.append(dct)
            result_dict[imp] = ls
        return result_dict

    @timeit
    def nan_click_viewed__deleting(self, X, y=None): # Функция удаления строк с пустыми значениями клика и с view == 0
        X = X[X['click'].isna() == False]
        X = X[X['view'] != 0]
        return X

    @timeit
    def dub_dropping(self, X, y=None): # Функция удаления дубликатов примеров
        X = X.drop_duplicates()
        return X




