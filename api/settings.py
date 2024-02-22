from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    status_url: str = '/status'
    version_url: str = '/version'
    
    PREP_TOOLS_DICT_PATH: str = '24news_practice/pickles/prep_tools_dict_tf.pkl'
    KERAS_MODEL_JSON: str = '24news_practice/pickles/tf_model.json'
    KERAS_MODEL_WIGHTS: str = '24news_practice/pickles/tf_weights.h5'
    CPC_WITH_WEIGHTS_PATH: str = '24news_practice/pickles/cpc_with_weight_only_creative_id.pkl'

    CLICKHOSE_DF_PATH: str = '24news_practice/internship/clickhouse.csv'
    CREATIVES_DF_PATH: str = '24news_practice/internship/creatives.csv'
    LEADS_DF_PATH: str = '24news_practice/internship/leads.csv'
    TEST_SPARSE_PATH: str = '24news_practice/pickles/x_test_prep_test.pkl'

settings = Settings()
