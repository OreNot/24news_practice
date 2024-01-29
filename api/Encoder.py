import dill
import logging

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/Encoder.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")


class Encoder:

    def __init__(self, encoder_pickle_path):
        with open(encoder_pickle_path, 'rb') as in_strm:
            preprocessing_dict = dill.load(in_strm)

        self.encoder = preprocessing_dict['encoder']
        self.req_df_columns = preprocessing_dict['req_df_columns']
        logging.info(f'Encoder loaded from  {encoder_pickle_path}')


