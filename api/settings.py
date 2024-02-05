from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    status_url: str = '/status'
    version_url: str = '/version'
    MODEL_PICKLE_PATH: str = '24news_practice/pickles/model_cat_dict_3.pkl'
    PREPROCESSOR_PICKLE_PATH: str = '24news_practice/pickles/preprocessor_cat_pipline_dict_3.pkl'
    ENCODER_PICKLE_PATH: str = '24news_practice/pickles/preprocessing_dict_cat_3.pkl'
    PREP_TOOLS_DICT_PATH: str = '24news_practice/pickles/prep_tools_dict.pkl'
    CLICKHOSE_DF_PATH: str = '24news_practice/internship/clickhouse.csv'
    CREATIVES_DF_PATH: str = '24news_practice/internship/creatives.csv'
    LEADS_DF_PATH: str = '24news_practice/internship/leads.csv'


settings = Settings()
