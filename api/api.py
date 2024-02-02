import gc
import logging
import warnings
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI
import sys
sys.path.insert(0, '.')

from .CPC import CPC
from .Encoder import Encoder
from .Model import Model
from .Prediction import Prediction
from .Preprocessor import Preprocessor
from .SSP import SSP
from .Status import Status
from .Tools import Tools
from .settings import settings

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/api.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

# Загрузка энкодера
ohe = Encoder(settings.ENCODER_PICKLE_PATH).encoder

# Загрузка препроцессора
preprocessor = Preprocessor(settings.PREPROCESSOR_PICKLE_PATH)
# Загрузка обученной модели
model = Model(settings.MODEL_PICKLE_PATH)

# Подсчёт CPC
cpc = CPC() # Ёмкая операция, стоит выполнять по распианию, хранить в бд уже вычисленное значение и здесь читать
cpc_df = cpc.CPC_calculating()

# Инструменты обработки данных
tools = Tools()
df_creatives = object

@app.get(settings.status_url) # Метод обработки запроса статуса api
async def status():
    return Status()

@app.get(settings.version_url) # Метод обработки запроса получения данных модели
def version():
    return model.model_metadata

@app.post('/predict', response_model = Prediction) # Метод предикта
def predict(ssp_req : SSP):

    # Определяем время получения запроса и преобразуем его в строку
    req_datetime = datetime.now()
    req_datetime_str = req_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # Логируем факт получения запроса
    logging.info(f'API request received at {req_datetime_str}')
    # Преобразовываем данные полученные в запросе в датафрейм и получаем датафрейм imps - creative_id
    cretive_tag_df, req_df = tools.get_creatives_imps_df(ssp_req, req_datetime_str)

    print(cretive_tag_df)
    print(req_df)

    # Мержим датафрейм запроса с креативами
    X_test = tools.with_creatives_mergding(req_df)
    X_test.drop('click', axis=1, inplace = True)
    # Обрабатываем пустые значения
    #df = tools.nan_filling(df)
    # Определяем X

    # Трансформируем X с помощью пайплайна препроцессора
    X_test_prep = preprocessor.preprocessor.transform(X_test)
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
    print(res)
    print(cretive_tag_df)
    res = pd.merge(res, cretive_tag_df, on="creative_id")

    del cretive_tag_df
    gc.collect()

    # На удаляем дубликаты
    res = res.drop_duplicates()

    # удаляем лишние параметры
    res = res[['imp_id', 'tagid', 'creative_id', 'CPM', 'plcmtcnt']]

    # Получаем список imp_id
    imps_list = res['imp_id'].unique()
    tag_list = res['tagid'].unique()
    print(res)
    rs = pd.DataFrame(columns = ['imp_id', 'tagid', 'creative_id', 'CPM'])
    # Выделяем самые прибыльные креативы в соответствии с plcmtcnt
    for imp in imps_list:

        temp = res[res['imp_id'] == imp]
        print(temp)
        tags = temp['tagid'].unique()
        print(tags)
        if temp.shape[0] > 0:

            temp = temp.nlargest(temp['plcmtcnt'].unique()[0], 'CPM')
            rs = pd.concat([rs, temp[['imp_id', 'tagid', 'creative_id', 'CPM']]], ignore_index=True)
            #res.drop(temp.index, axis=0, inplace=True, errors = 'ignore')
            for creative_id in temp['creative_id'].unique():
                res.drop(res.loc[(res['creative_id'] == creative_id) & (res['tagid'] == tags[0])].index, inplace=True)

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


if __name__ == '__api__':
      uvicorn.run("api:app", host="127.0.0.1", port=9000, reload=True)






