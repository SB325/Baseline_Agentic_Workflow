from typing import Optional
from pydantic import BaseModel

class ObservationConfigVals(BaseModel):
    val: Optional[int] = 1

class ObservationConfig(BaseModel):
    observation_config_vals: Optional[ObservationConfigVals] = None

class RootConfig(BaseModel):
    observation: Optional[ObservationConfig] = None