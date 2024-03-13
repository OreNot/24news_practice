import warnings
import subprocess
import pandas as pd
import sys
import uvicorn
import logging.config
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
from fastapi import FastAPI, Request
from .CPC import CPC
from .SSP import SSP
from .Status import Status
from .Tools import Tools
from .Creatives import Creatives
from .settings import settings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from ..config.logging_config import dict_config


logging.config.dictConfig(dict_config)

api_logger = logging.getLogger('api_logger')
api_logger.setLevel(logging.INFO)

app = FastAPI()

cpc_df = CPC.CPC_calculating()

# РРЅСЃС‚СЂСѓРјРµРЅС‚С‹ РѕР±СЂР°Р±РѕС‚РєРё РґР°РЅРЅС‹С…
tools = Tools()
model = tools.model
model_metadata = tools.model_metadata
preprocessor = tools.preprocessor

df_creatives = Creatives().df_creatives

@app.get(settings.status_url)  # РњРµС‚РѕРґ РѕР±СЂР°Р±РѕС‚РєРё Р·Р°РїСЂРѕСЃР° СЃС‚Р°С‚СѓСЃР° api
async def status(request: Request):
    api_logger.info(f'Status check request has been received from {request.client.host}')
    return Status()

@app.get(settings.version_url)  # РњРµС‚РѕРґ РѕР±СЂР°Р±РѕС‚РєРё Р·Р°РїСЂРѕСЃР° РїРѕР»СѓС‡РµРЅРёСЏ РґР°РЅРЅС‹С… РјРѕРґРµР»Рё
def version(request: Request):
    api_logger.info(f'Get version request has been received from {request.client.host}')
    return model_metadata

@app.get(settings.uptime_url)
def uptime(request: Request):
    test = subprocess.Popen(["uptime", "-p"], stdout=subprocess.PIPE)
    UPTIME = str(test.communicate()[0]).replace("b'up ", '').replace("\\n'", '')
    api_logger.info(f'Get uptime request has been received from {request.client.host}')
    return f'Current uptime is {UPTIME}'

@app.post('/predict')  # РњРµС‚РѕРґ РїСЂРµРґРёРєС‚Р°
async def predict(ssp_req: SSP, request: Request):
    api_logger.info(f'Predict request has been received from {request.client.host}')
    predict_start_time = datetime.now()
    # Определяем время получения запроса и преобразуем его в строку
    req_datetime_str = predict_start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Преобразовываем данные полученные в запросе в датафрейм и получаем датафрейм imps - creative_id
    cretive_tag_df, req_df = tools.get_creatives_imps_df(ssp_req, req_datetime_str)

    res = req_df.merge(df_creatives, on='creative_id')
    res.fillna(0, inplace=True)


    x_test_prep = preprocessor.transform(res)

    # Добавляем параметр CTR к X
    res['CTR'] = model.predict(x_test_prep, verbose=0)[:, 1]
    api_logger.info(f'CTR predicted successfully')
    # Удаляем избыточные параметры
    res = res[['creative_id', 'tag_id', 'CTR']]

    # Мержим с датафреймом imps - creative_id, для ассоциации с imp_id
    res = pd.merge(res, cretive_tag_df, on=['creative_id', 'tag_id'])
    # Мержим CTR с CPC
    res = res.merge(cpc_df, on=['creative_id'])
    
    res.drop_duplicates(inplace=True)
    
    # Рассчитываем CPM
    res['CPM'] = (res['CTR'] * res['click_profit'] * 1000)
    api_logger.info(f'CPM calculated successfully')
    
    # удаляем лишние параметры
    res = res[['imp_id', 'tag_id', 'creative_id', 'CPM', 'plcmtcnt', 'creatives_list_id']]

    res.drop_duplicates(inplace=True)

    rs_list = []


    # Выделяем самые прибыльные креативы в соответствии с plcmtcnt
    for imp in res['imp_id'].unique():

        temp = res[res['imp_id'] == imp]
        drop_list = []
        if temp.shape[0] > 0:
            temp = temp.nlargest(temp['plcmtcnt'].unique()[0], 'CPM')

            temp_list = temp[['imp_id', 'tag_id', 'creative_id', 'CPM']].values.tolist()

            rs_list.extend(temp_list)

            for creative_id in temp['creative_id'].unique():
                drop_list.extend(res.loc[(res['creatives_list_id'] == temp['creatives_list_id'].unique()[0]) & (
                        res['creative_id'] == creative_id) & (res['tag_id'] == temp['tag_id'].unique()[0])].index)

            res.drop(drop_list, inplace=True)

    
    # Логируем
    api_logger.info(f'Response created successfully, predict time = {(datetime.now() - predict_start_time)}')
    # Отправляем ответ клиенту
    return tools.get_result_dict(res['imp_id'].unique(),
                                 pd.DataFrame(rs_list, columns=['imp_id', 'tag_id', 'creative_id', 'CPM']))
def start_uvicorn():
    uvicorn.run('practice.api.api:app', host=settings.server_ip, port=settings.server_port, reload=True)
    api_logger.info(f'Uvicorn started successfully')
