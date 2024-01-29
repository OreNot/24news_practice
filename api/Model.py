import dill
import logging

logging.basicConfig(level=logging.INFO, filename="../logs/Model.log", filemode="a", format="%(asctime)s %(levelname)s %(message)s")
class Model:

    def __init__(self, model_picle_path):

        with open(model_picle_path, 'rb') as in_strm:
            model_dict = dill.load(in_strm)

        self.model = model_dict['model']
        self.model_metadata = model_dict['metadata']
        logging.info(f'Model {self.model_metadata} from {model_picle_path} loaded')



