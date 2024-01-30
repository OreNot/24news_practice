import pandas as pd
import numpy as np
from .settings import settings
import logging

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/Tools.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")

class Tools:

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

    def with_creatives_megding(self, X): # Функция мержа с датафреймом креативов

        try:
            df_creatives = pd.read_csv(settings.CREATIVES_DF_PATH)
            logging.info(f'Creatives dataframe read')
        except Exception as e:
            logging.error(f'Creatives dataframe read error: {e}')

        df_creatives = df_creatives.rename(columns={'id': 'creative_id'})
        df = pd.merge(X, df_creatives, on="creative_id")
        return df
    def get_creatives_imps_df(self, ssp_req, req_df_columns,req_datetime_str): # Функция получения датафрейма imps - creative_id и датафрейма запроса из запроса

        X = pd.DataFrame(columns=['imp_id', 'plcmtcnt', 'creatives_list_id', 'creative_id'])
        y = pd.DataFrame(columns = req_df_columns)

        for key, value in ssp_req.creatives_list.items():
            # print(value)
            imp_id = ''
            for imp in ssp_req.imps:
                if imp['creatives_list_id'] == key:
                    imp_id = imp['id']
                    plcmtcnt = imp['plcmtcnt']
                    for list in value:
                        for key_, value_ in list.items():
                            new_row_X = {'imp_id': imp_id,
                                       'plcmtcnt': plcmtcnt,
                                       'creative_id': key_,
                                       'creatives_list_id': key,
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
                                       'creative_id': key_,
                                       'event_date_time': req_datetime_str

                                       }
                            y.loc[len(y)] = new_row_y

        y = y.fillna(0)
        return X, y

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

    def nan_click_viewed__deleting(self, X, y=None): # Функция удаления строк с пустыми значениями клика и с view == 0
        X = X[X['click'].isna() == False]
        X = X[X['view'] != 0]
        return X

    def dub_dropping(self, X, y=None): # Функция удаления дубликатов примеров
        X = X.drop_duplicates()
        return X




