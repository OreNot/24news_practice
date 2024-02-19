import logging, warnings, pandas as pd, sys, numpy as np, dill
from scipy.sparse import hstack
from datetime import datetime
from fastapi import FastAPI
from .CPC import CPC
from .SSP import SSP
from .Status import Status
from .Tools import Tools, timeit
from .Creatives import Creatives
from .settings import settings

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/api.log",
                    filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

# Подсчёт CPC
# cpc = CPC()  # Ёмкая операция, стоит выполнять по распианию, хранить в бд уже вычисленное значение и здесь читать
#cpc_df = CPC.CPC_calculating()

with open(settings.CPC_WITH_WEIGHTS_PATH, 'rb') as in_strm:
    cpc_dict = dill.load(in_strm)

cpc_df = cpc_dict['cpc_df']



# Инструменты обработки данных
tools = Tools()
model = tools.model
model_metadata = tools.model_metadata
# encoder = tools.encoder
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
    req_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Логируем факт получения запроса
    logging.info(f'API request received at {req_datetime_str}')

    # Преобразовываем данные полученные в запросе в датафрейм и получаем датафрейм imps - creative_id
    cretive_tag_df, req_df = tools.get_creatives_imps_df(ssp_req, req_datetime_str)
    # print(f'cretive_tag_df: {cretive_tag_df}')
    df = req_df.merge(df_creatives, on='creative_id')
    df.fillna(0, inplace=True)

    print(f'df: {df}')

    x_test_prep = preprocessor.transform(df)
    df['CTR'] = model.predict(x_test_prep, verbose=0)[:, 1]
    # print(f'df: {df}')
    # print(f'df: {df.columns}')
    df.drop(['impression_id', 'auction_date_time', 'impression_hash', 'ssp', 'auction_id',
       'bid_id', 'auction_type', 'bid_floor', 'bid_price',
       'loss_reason', 'is_win', 'pay_price', 'is_pay', 'view', 'place_number',
       'ssp_user_id', 'campaign_id_x', 'stream_id_x', 'link_id',
       'format', 'device', 'OS', 'browser', 'geo_country', 'geo_city', 'ip_v4',
       'ip_v6', 'site_id', 'iab_category_x', 'ab_test',
       'enter_utm_source', 'enter_utm_campaign', 'enter_utm_medium',
       'enter_utm_content', 'enter_utm_term', 'event_date_time', 'status',
       'is_deleted', 'campaign_id_y', 'user_id', 'stream_id_y', 'theme',
       'second_theme', 'iab_category_y', 'image', 'image_extension',
       'mime_type', 'image_tag', 'created_at', 'updated_at'], axis = 1, inplace = True)

    df.sort_values(by='CTR', ascending=False, inplace=True)
    df.drop_duplicates(subset=['creative_id', 'tag_id'], keep='first', inplace=True)
    print(f'df: {df}')
    # print(f'df: {df.columns}')


    print(f'cretive_tag_df: {cretive_tag_df}')

    df = pd.merge(df, cretive_tag_df, on=['creative_id', 'tag_id'])
    df.sort_values(by='imp_id')
    print(f'df: {df}')
    print(f"df: {df['imp_id'].unique()}")

    print(f'cpc_df: {cpc_df}')

    df = df.merge(cpc_df, on=['creative_id', 'tag_id'])
    print(df)
    print(f"{df['imp_id'].unique()}")



    return ''