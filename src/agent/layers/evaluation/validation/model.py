from typing import Optional
from pydantic import BaseModel

class EvalTestConfigVals(BaseModel):
    val: Optional[int] = 1

class TestConfig(BaseModel):
    eval_test_config_vals: Optional[EvalTestConfigVals] = None

class RootConfig(BaseModel):
    test: Optional[TestConfig] = None