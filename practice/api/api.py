import logging, warnings, pandas as pd, sys, uvicorn
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



logging.basicConfig(level=logging.INFO, filename="practice/logs/api.log",
                    filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

cpc_df = CPC.CPC_calculating()

# Инструменты обработки данных
tools = Tools()
model = tools.model
model_metadata = tools.model_metadata
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
     # ���������� ����� ��������� ������� � ����������� ��� � ������
    req_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # �������� ���� ��������� �������
    logging.info(f'API request received at {req_datetime_str}')

    # ��������������� ������ ���������� � ������� � ��������� � �������� ��������� imps - creative_id
    cretive_tag_df, req_df = tools.get_creatives_imps_df(ssp_req, req_datetime_str)

    res = req_df.merge(df_creatives, on='creative_id')
    res.fillna(0, inplace=True)


    x_test_prep = preprocessor.transform(res)

    # ��������� �������� CTR � X
    res['CTR'] = model.predict(x_test_prep, verbose=0)[:, 1]
  
    # ������� ���������� ���������
    res = res[['creative_id', 'tag_id', 'CTR']]

    # ������ � ����������� imps - creative_id, ��� ���������� � imp_id
    res = pd.merge(res, cretive_tag_df, on=['creative_id', 'tag_id'])
    # ������ CTR � CPC
    res = res.merge(cpc_df, on=['creative_id'])
    
    res.drop_duplicates(inplace=True)
    
    # ������������ CPM
    res['CPM'] = (res['CTR'] * res['click_profit'] * 1000)
    
    #print(res.sort_values('CPM', ascending=False))

    # ������� ������ ���������
    res = res[['imp_id', 'tag_id', 'creative_id', 'CPM', 'plcmtcnt', 'creatives_list_id']]

    res.drop_duplicates(inplace=True)

    rs_list = []


    # �������� ����� ���������� �������� � ������������ � plcmtcnt
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

    # ��������� ����� ��������� �������
    res_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # ��������
    logging.info(f'API request processed at {res_datetime_str}')
    # ���������� ����� �������
    return tools.get_result_dict(res['imp_id'].unique(),
                                 pd.DataFrame(rs_list, columns=['imp_id', 'tag_id', 'creative_id', 'CPM']))
def start_uvicorn():
    uvicorn.run('practice.api.api:app', host=settings.server_ip, port=settings.server_port, reload=True)
