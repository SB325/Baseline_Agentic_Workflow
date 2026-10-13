from pydantic import BaseModel


class PolicyConfigVals(BaseModel):
  val: int = 1


class Policy(BaseModel):
  policy_config_vals: PolicyConfigVals = PolicyConfigVals()


class ConfigModel(BaseModel):
  policy: Policy