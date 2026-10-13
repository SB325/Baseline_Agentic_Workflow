from typing import Optional
from pydantic import BaseModel


class Connection(BaseModel):
  vllm_address: Optional[str] = None
  client_id: Optional[str] = None
  system_prompt: Optional[str] = None


class Inference(BaseModel):
  max_tokens: int = 256
  top_p: float = 0.9
  top_k: int = 20
  repetition_penalty: float = 1.1
  presence_penalty: float = 0.1
  frequency_penalty: float = 0.1
  remember: bool = False


class LLMConfig(BaseModel):
  connection: Connection = Connection()
  inference: Inference = Inference()


class ConfigModel(BaseModel):
  llm: LLMConfig