from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    status_url: str = '/status'
    version_url: str = '/version'
    server_ip: str = '192.168.1.173'
    server_port: int = 9000
    
    PREP_TOOLS_DICT_PATH: str = 'practice/pickles/prep_tools_dict_tf.pkl'
    KERAS_MODEL_JSON: str = 'practice/pickles/tf_model.json'
    KERAS_MODEL_WIGHTS: str = 'practice/pickles/tf_weights.h5'
    CPC_WITH_WEIGHTS_PATH: str = 'practice/pickles/cpc_with_weight_only_creative_id.pkl'

    CLICKHOSE_DF_PATH: str = 'practice/internship/clickhouse.csv'
    CREATIVES_DF_PATH: str = 'practice/internship/creatives.csv'
    LEADS_DF_PATH: str = 'practice/internship/leads.csv'
    TEST_SPARSE_PATH: str = 'practice/pickles/x_test_prep_test.pkl'

settings = Settings()
