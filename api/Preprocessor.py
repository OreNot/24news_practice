import dill
import logging

logging.basicConfig(level=logging.INFO, filename="24news_practice/logs/Preprocessor.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")
class Preprocessor:
    def __init__(self, preprocessor_pickle_path):
        with open(preprocessor_pickle_path, 'rb') as in_strm:
            preprocessor_pipline_dict = dill.load(in_strm)

        self.preprocessor = preprocessor_pipline_dict['preprocessor']

        logging.info(f'Preprocessor loaded from  {preprocessor_pickle_path}')


