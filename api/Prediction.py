from pydantic import BaseModel
from typing import Dict, List


class Prediction(BaseModel):
    Result: Dict[str, List[Dict[str, float]]]