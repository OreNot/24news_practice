import pandas as pd
import gc
from .settings import settings
import logging

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/CPC_.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")

DF_CLICKHOUSE = object
DF_CREATIVES = object
DF_LEADS = object


def nan_viewed__deleting(X, y=None): # Функция удаления не показанных креативов
    X = X[X['view'] != 0]
    return X
class CPC_: # Класс для расчёта CPC

    def CPC_calculating(self): # Функция расчета CPC
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

        DF_CREATIVES = DF_CREATIVES.rename(columns={'id': 'creative_id'})

        # Объединение датасетов
        temp_df = pd.merge(DF_CLICKHOUSE, DF_CREATIVES, on="creative_id")
        del DF_CLICKHOUSE
        del DF_CREATIVES
        gc.collect()

        temp_df = nan_viewed__deleting(temp_df)
        # аггрегация среативов по сумме кликов
        temp_df = temp_df.groupby(['creative_id'])['click'].agg('sum')

        temp_df = pd.DataFrame({'creative_id': temp_df.index, 'click_count': temp_df.values})
        # Удаление дубликатов
        temp_df = temp_df.drop_duplicates()
        # Сокращение датасета leads
        temp_df_leads = DF_LEADS[['creative_id', 'profit']]
        del DF_LEADS
        gc.collect()
        # Аггрегация датасета leads по сумме заказов для каждого creative_id
        temp_df_leads = temp_df_leads.groupby(['creative_id'])['profit'].agg('sum')

        temp_df_leads = pd.DataFrame({'creative_id': temp_df_leads.index, 'profit_sum': temp_df_leads.values})
        # Объединение датасетов
        result_df = pd.merge(temp_df, temp_df_leads, on="creative_id")
        # Расчёт CPC
        result_df['click_profit'] = result_df['profit_sum'] / result_df['click_count']

        return result_df