import gc
from typing import List, Dict

from fastapi import FastAPI
from .settings import settings
from .Preprocessor import Preprocessor
from .Model import Model
from .Encoder import Encoder
from .CPC import CPC
from .SSP import SSP
from .Tools import Tools
from .Status import Status
from .Prediction import Prediction
import pandas as pd
import numpy as np
from datetime import datetime
import logging

import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, filename="../logs/api.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

# Загрузка энкодера
ohe = Encoder(settings.ENCODER_PICKLE_PATH).encoder
# Загрузка списка параметров объединенного датасета
req_df_columns = Encoder(settings.ENCODER_PICKLE_PATH).req_df_columns
# Загрузка препроцессора
preprocessor = Preprocessor(settings.PREPROCESSOR_PICKLE_PATH)
# Загрузка обученной модели
model = Model(settings.MODEL_PICKLE_PATH)

# Подсчёт CPC
cpc = CPC() # Ёмкая операция, стоит выполнять по распианию, хранить в бд уже вычисленное значение и здесь читать
cpc_df = cpc.CPC_calculating()

# Инструменты обработки данных
tools = Tools()

@app.get(settings.status_url) # Метод обработки запроса статуса api
async def status():
    return Status()

@app.get(settings.version_url) # Метод обработки запроса получения данных модели
def version():
    return model.model_metadata

@app.post('/predict', response_model = Prediction) # Метод предикта
async def predict(ssp_req : SSP):

    # Определяем время получения запроса и преобразуем его в строку
    req_datetime = datetime.now()
    req_datetime_str = req_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Логируем факт получения запроса
    logging.info(f'API request received at {req_datetime_str}')
    # Преобразовываем данные полученные в запросе в датафрейм и получаем датафрейм imps - creative_id
    cretive_imp_df, req_df = tools.get_creatives_imps_df(ssp_req, req_df_columns, req_datetime_str)
    # Мержим датафрейм запроса с креативами
    df = tools.with_creatives_megding(req_df)
    # Обрабатываем пустые значения
    df = tools.nan_filling(df)
    # Определяем X
    X_test = df.drop('click', axis=1)
    # Трансформируем X с помощью пайплайна препроцессора
    X_test_prep = preprocessor.preprocessor.fit_transform(X_test)
    # Предсказываем вероятности кликов по креативам из запроса
    probs = model.model.predict_proba(X_test_prep.toarray())
    # Добавляем параметр CTR к X
    X_test['CTR'] = probs[:, 1]
    # Удаляем избыточные параметры
    res = X_test[['creative_id', 'CTR']]

    del X_test
    gc.collect()

    # Мержим CTR с CPC
    res = pd.merge(res, cpc_df, on="creative_id")
    # Рассчитываем CPM
    res['CPM'] = (res['CTR'] * res['click_profit'] * 100)
    # Мержим с датафреймом imps - creative_id, для ассоциации с imp_id
    res = pd.merge(res, cretive_imp_df, on="creative_id")
    # На удаляем дубликаты
    res = res.drop_duplicates()
    # удаляем лишние параметры
    res = res[['imp_id', 'creative_id', 'CPM']]
    # Получаем список imp_id
    imps_list = res['imp_id'].unique()
    # Формируем результирующий словарь
    res_dict = tools.get_result_dict(imps_list, res)

    del res
    gc.collect()
    # Фиксируем время обработки запроса
    res_datetime = datetime.now()
    res_datetime_str = res_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Логируем
    logging.info(f'API request processed at {res_datetime_str}')
    # Отправляем ответ клиенту
    return {'Result': res_dict}








