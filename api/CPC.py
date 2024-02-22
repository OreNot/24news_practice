import pandas as pd
import gc
import datetime
from .settings import settings
import logging

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/CPC.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")

DF_CLICKHOUSE = object
DF_CREATIVES = object
DF_LEADS = object


def nan_viewed__deleting(X, y=None):  # Р¤СѓРЅРєС†РёСЏ СѓРґР°Р»РµРЅРёСЏ РЅРµ РїРѕРєР°Р·Р°РЅРЅС‹С… РєСЂРµР°С‚РёРІРѕРІ
    X = X[X['view'] != 0]
    return X
class CPC:  # РљР»Р°СЃСЃ РґР»СЏ СЂР°СЃС‡С‘С‚Р° CPC

    @staticmethod
    def CPC_calculating():  # Р¤СѓРЅРєС†РёСЏ СЂР°СЃС‡РµС‚Р° CPC
        # Считывание датасетов
        try:
            DF_CLICKHOUSE = pd.read_csv(settings.CLICKHOSE_DF_PATH)
            logging.info(f'Clickhouse dataframe read')
        except Exception as e:
            logging.error(f'Clickhouse dataframe read error: {e}')

        try:
            DF_CREATIVES = pd.read_csv(settings.CREATIVES_DF_PATH)
            logging.info(f'Creatives dataframe read')
        except Exception as e:
            logging.error(f'Creatives dataframe read error: {e}')

        try:
            DF_LEADS = pd.read_csv(settings.LEADS_DF_PATH)
            logging.info(f'Leads dataframe read')
        except Exception as e:
            logging.error(f'Leads dataframe read error: {e}')

        # Удаляем лишние признаки
        DF_CLICKHOUSE.drop(
            ['impression_hash', 'ssp', 'auction_id', 'bid_id', 'auction_type', 'bid_floor', 'bid_price', 'is_win',
             'pay_price', 'is_pay', 'ssp_user_id', 'campaign_id', 'stream_id', 'link_id', 'ip_v4', 'ip_v6', 'site_id',
             'iab_category', 'ab_test', 'enter_utm_source', 'enter_utm_campaign', 'enter_utm_medium',
             'enter_utm_content', 'enter_utm_term', 'auction_date_time'], axis=1, inplace=True)

        DF_CREATIVES.rename(columns={'id': 'creative_id'}, inplace=True)

        # Объединение датасетов
        temp_df = pd.merge(DF_CLICKHOUSE, DF_CREATIVES, on=['creative_id'])

        del DF_CLICKHOUSE
        del DF_CREATIVES
        gc.collect()

        temp_df = nan_viewed__deleting(temp_df)

        # Удаляем избыточные признаки
        temp_df.drop(
            ['tag_id', 'event_date_time', 'impression_id', 'loss_reason', 'view', 'place_number', 'format', 'device',
             'OS', 'browser', 'geo_country', 'geo_city', 'status', 'is_deleted', 'campaign_id', 'user_id', 'stream_id',
             'theme', 'second_theme', 'image', 'iab_category', 'image_extension', 'mime_type', 'image_tag',
             'created_at', 'updated_at'], axis=1, inplace=True)

        # аггрегация среативов по сумме кликов
        temp_df = temp_df.groupby(['creative_id'])['click'].agg('sum').reset_index()
        temp_df = temp_df[temp_df['click'] != 0]


        # Удаление дубликатов
        temp_df = temp_df.drop_duplicates()
        #print(f'temp_df:\n{temp_df}')

        DF_LEADS = DF_LEADS[DF_LEADS['order_status'] == 1]
        # Преобразуем дату последнего обновления информации в формат даты
        DF_LEADS['updated_at'] = pd.to_datetime(DF_LEADS['updated_at'], format='%Y-%m-%d %H:%M:%S').dt.date
        DF_LEADS['updated_at'] = pd.to_datetime(DF_LEADS['updated_at'], format='%Y-%m-%d')

        temp_df_leads = DF_LEADS[['creative_id', 'profit', 'updated_at']]

        temp_df_leads.drop_duplicates(inplace=True)

        temp_df_leads = temp_df_leads.groupby(['creative_id', 'profit'])['updated_at'].agg('max').reset_index()

        now = datetime.datetime.now()
        temp_df_leads['days_ago'] = (now - temp_df_leads['updated_at']).dt.days

        # Принимаем самое большое количество дней с текущей даты за 1 и считаем коэффициенты, ориентируясь на степень устаревания информации
        max_days = temp_df_leads['days_ago'].max()
        temp_df_leads['coef'] = 1 - ((temp_df_leads['days_ago'] / max_days))

        # Считаем профит с учетом коэффициента и аггрегируем по среднему значению коэффициента
        temp_df_leads['profit_by_coef'] = temp_df_leads['profit'] * temp_df_leads['coef']
        temp_df_leads = temp_df_leads[['creative_id', 'profit_by_coef']]
        temp_df_leads = temp_df_leads.groupby(['creative_id'])['profit_by_coef'].agg('mean').reset_index()

        # Мержим лиды и клики
        df = pd.merge(temp_df, temp_df_leads, on=['creative_id'])

        df['click_profit'] = df['profit_by_coef'] / df['click']

        result_df = df.drop(['click', 'profit_by_coef'], axis = 1)

        return result_df