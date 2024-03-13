import pandas as pd
from .settings import settings
from ..config.logging_config import dict_config
import logging.config

logging.config.dictConfig(dict_config)

creative_logger = logging.getLogger('creative_logger')
creative_logger.setLevel(logging.INFO)

class Creatives:

        def __init__(self):

        	try:
            		self.df_creatives = pd.read_csv(settings.CREATIVES_DF_PATH)
            		self.df_creatives = self.df_creatives.rename(columns={'id': 'creative_id'})
            		self.df_creatives.drop(['campaign_id', 'stream_id', 'iab_category'], axis = 1, inplace = True)
            		creative_logger.info(f'Creatives dataframe read')
        	except Exception as e:
            		creative_logger.error(f'Creatives dataframe read error: {e}')

        	creative_logger.info(f'Dataframe Creatives loaded from {settings.CREATIVES_DF_PATH}')