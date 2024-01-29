from pydantic import BaseModel
from typing import List, Dict, Union, Any

class SSP(BaseModel):
    user_id: str
    ip: str
    site_id: str
    os: str # OS
    browser: str
    device: int
    country: str # geo_country
    city: int # geo_city
    news_category: Union[int, str] # loss_reason
    us: str # enter_utm_source
    ucm: str # enter_utm_campaign
    um: str # enter_utm_medium
    uct: str # enter_utm_content
    ut: str # enter_utm_term
    creatives_list: Dict[str, List[Dict[str, int]]]
    imps: List[Dict[str, Any]]
