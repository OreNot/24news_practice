import gc
import logging
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
import sys
sys.path.insert(0, '.')

from .CPC import CPC
from .Prediction import Prediction
from .SSP import SSP
from .Status import Status
from .Tools import Tools
from .settings import settings

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/api_2.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

# Подсчёт CPC
cpc = CPC() # Ёмкая операция, стоит выполнять по распианию, хранить в бд уже вычисленное значение и здесь читать
cpc_df = cpc.CPC_calculating()

# Инструменты обработки данных
tools = Tools()
model = tools.model
model_metadata = tools.model_metadata
encoder = tools.encoder
preprocessor = tools.preprocessor

df_creatives = object

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

@app.get(settings.status_url) # Метод обработки запроса статуса api
async def status():
    return Status()

@app.get(settings.version_url) # Метод обработки запроса получения данных модели
def version():
    return model_metadata

@app.post('/predict', response_model = Prediction) # Метод предикта
def predict(ssp_req : SSP):
    # Определяем время получения запроса и преобразуем его в строку
    req_datetime = datetime.now()
    req_datetime_str = req_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Логируем факт получения запроса
    logging.info(f'API request received at {req_datetime_str}')
    # Преобразовываем данные полученные в запросе в датафрейм и получаем датафрейм imps - creative_id
    cretive_tag_df, req_df = tools.get_creatives_imps_df(ssp_req, req_datetime_str)

    X_test = tools.with_creatives_mergding(req_df)
    X_test.fillna(0, inplace = True)

    X_test_prep = preprocessor.transform(X_test)

    probs = model.predict_proba(X_test_prep.toarray())
    # Добавляем параметр CTR к X
    X_test['CTR'] = probs[:, 1]
    # Удаляем избыточные параметры
    res = X_test[['creative_id', 'tag_id', 'CTR']]

    # Мержим CTR с CPC
    res = res.merge(cpc_df, on=['creative_id'])
    # Рассчитываем CPM
    res['CPM'] = (res['CTR'] * res['click_profit'] * 100)
    # Мержим с датафреймом imps - creative_id, для ассоциации с imp_id
    res = pd.merge(res, cretive_tag_df, on=['creative_id'])

    # удаляем лишние параметры
    res = res[['imp_id', 'tag_id', 'creative_id', 'CPM', 'plcmtcnt']]

    # Получаем список imp_id
    imps_list = res['imp_id'].unique()

    rs = pd.DataFrame(columns=['imp_id', 'tag_id', 'creative_id', 'CPM'])
    # Выделяем самые прибыльные креативы в соответствии с plcmtcnt
    for imp in imps_list:

        temp = res[res['imp_id'] == imp]
        temp.sort_values(by = 'CPM', ascending = False, inplace = True)
        temp.drop_duplicates(subset = ['imp_id', 'tag_id', 'creative_id'], keep = 'first', inplace = True)
        tags = temp['tag_id'].unique()
        if temp.shape[0] > 0:

            temp = temp.nlargest(temp['plcmtcnt'].unique()[0], 'CPM')
            rs = pd.concat([rs, temp[['imp_id', 'tag_id', 'creative_id', 'CPM']]], ignore_index=True)

            for creative_id in temp['creative_id'].unique():
                res.drop(res.loc[(res['creative_id'] == creative_id) & (res['tag_id'] == tags[0])].index, inplace=True)

    # Формируем результирующий словарь
    res_dict = tools.get_result_dict(imps_list, rs)

    del res
    gc.collect()

    # Фиксируем время обработки запроса
    res_datetime = datetime.now()
    res_datetime_str = res_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Логируем
    logging.info(f'API request processed at {res_datetime_str}')
    # Отправляем ответ клиенту
    return {'Result': res_dict}


