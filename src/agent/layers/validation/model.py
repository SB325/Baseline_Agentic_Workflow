from pydantic import BaseModel


class HarnessConfigVals(BaseModel):
  val: int


class Harness(BaseModel):
  harness_config_vals: HarnessConfigVals


class ConfigModel(BaseModel):
  harness: Harness