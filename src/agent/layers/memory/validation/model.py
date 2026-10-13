from pydantic import BaseModel


class ContextDetails(BaseModel):
  context_window_limit: int = 500


class SessionsDetails(BaseModel):
  session_id_prefix: str = "default_"


class ContextConfig(BaseModel):
  context: ContextDetails = ContextDetails()
  sessions: SessionsDetails = SessionsDetails()


class ConfigModel(BaseModel):
  context: ContextConfig