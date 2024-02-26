import pandas as pd
from .settings import settings
import logging

logging.basicConfig(level=logging.INFO, filename="practice/logs/Creatives.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")
class Creatives:

        def __init__(self):

        	try:
            		self.df_creatives = pd.read_csv(settings.CREATIVES_DF_PATH)
            		self.df_creatives = self.df_creatives.rename(columns={'id': 'creative_id'})
            		self.df_creatives.drop(['campaign_id', 'stream_id', 'iab_category'], axis = 1, inplace = True)
            		logging.info(f'Creatives dataframe read')
        	except Exception as e:
            		logging.error(f'Creatives dataframe read error: {e}')

        	logging.info(f'Dataframe Creatives loaded from {settings.CREATIVES_DF_PATH}')