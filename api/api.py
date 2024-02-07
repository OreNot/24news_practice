import logging, warnings, pandas as pd, sys
from datetime import datetime
from fastapi import FastAPI
from .CPC import CPC
from .SSP import SSP
from .Status import Status
from .Tools import Tools
from .Creatives import Creatives
from .settings import settings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/api.log",
                    filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

# Подсчёт CPC
#cpc = CPC()  # Ёмкая операция, стоит выполнять по распианию, хранить в бд уже вычисленное значение и здесь читать
cpc_df = CPC.CPC_calculating()

# Инструменты обработки данных
tools = Tools()
model = tools.model
model_metadata = tools.model_metadata
encoder = tools.encoder
preprocessor = tools.preprocessor

df_creatives = Creatives().df_creatives

@app.get(settings.status_url)  # Метод обработки запроса статуса api
async def status():
    return Status()

@app.get(settings.version_url)  # Метод обработки запроса получения данных модели
def version():
    return model_metadata

@app.post('/predict')  # Метод предикта
async def predict(ssp_req: SSP):
    # Определяем время получения запроса и преобразуем его в строку
    req_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Логируем факт получения запроса
    logging.info(f'API request received at {req_datetime_str}')
    # Преобразовываем данные полученные в запросе в датафрейм и получаем датафрейм imps - creative_id
    cretive_tag_df, req_df = tools.get_creatives_imps_df(ssp_req, req_datetime_str)

    res = req_df.merge(df_creatives, on='creative_id')
    res.fillna(0, inplace=True)

    x_test_prep = preprocessor.transform(res)

    res['CTR'] = model.predict_proba(x_test_prep.toarray())[:, 1]
    # Добавляем параметр CTR к X

    # Удаляем избыточные параметры
    res = res[['creative_id', 'tag_id', 'CTR']]

    # Мержим CTR с CPC
    res = res.merge(cpc_df, on=['creative_id'])
    # Рассчитываем CPM
    res['CPM'] = (res['CTR'] * res['click_profit'] * 100)
    # Мержим с датафреймом imps - creative_id, для ассоциации с imp_id
    res = pd.merge(res, cretive_tag_df, on=['creative_id', 'tag_id'])

    # удаляем лишние параметры
    res = res[['imp_id', 'tag_id', 'creative_id', 'CPM', 'plcmtcnt', 'creatives_list_id']]

    res.drop_duplicates(inplace=True)

    rs_list = []

    # Выделяем самые прибыльные креативы в соответствии с plcmtcnt
    for imp in res['imp_id'].unique():

        temp = res[res['imp_id'] == imp]

        if temp.shape[0] > 0:
            temp = temp.nlargest(temp['plcmtcnt'].unique()[0], 'CPM')

            temp_list = temp[['imp_id', 'tag_id', 'creative_id', 'CPM']].values.tolist()

            rs_list.extend(temp_list)

            for creative_id in temp['creative_id'].unique():
                res.drop(res.loc[(res['creatives_list_id'] == temp['creatives_list_id'].unique()[0]) & (res['creative_id'] == creative_id) & (res['tag_id'] == temp['tag_id'].unique()[0])].index, inplace=True)

    # Фиксируем время обработки запроса

    res_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Логируем
    logging.info(f'API request processed at {res_datetime_str}')
    # Отправляем ответ клиенту
    #return {'Result': res_dict}
    return tools.get_result_dict(res['imp_id'].unique(), pd.DataFrame(rs_list, columns=['imp_id', 'tag_id', 'creative_id', 'CPM']))

