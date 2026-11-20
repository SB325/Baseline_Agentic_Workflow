"""
Abstract base interfaces (Protocols) for the Agentic Baseline layers.
Concrete implementations are injected into BaseAgent at runtime.
"""
from typing import Protocol, runtime_checkable, Optional, Any, Dict, List


@runtime_checkable
class ILLM(Protocol):
    async def create_session(self, client_id: str, system_prompt: str) -> None: ...

    async def inference(
        self,
        client_id: str,
        prompt_string: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        remember: Optional[bool] = None,
    ) -> Dict[str, Any]: ...

    async def delete_session(self, client_id: str) -> None: ...


@runtime_checkable
class IMemory(Protocol):
    async def check_against_token_limit(
        self, string: str, verbose: bool = False
    ) -> bool: ...


@runtime_checkable
class IGuardrail(Protocol):
    async def check_input(self, text: str, context: Dict[str, Any]) -> bool: ...

    async def check_output(self, text: str, context: Dict[str, Any]) -> bool: ...


@runtime_checkable
class IAugmentation(Protocol):
    async def tool_connect(self) -> None: ...

    async def tool_disconnect(self) -> None: ...

    async def show_tools(self) -> List[Dict[str, Any]]: ...

    async def validate_llm_response(self, llm_response: Any) -> Dict[str, Any]: ...


@runtime_checkable
class IEvaluation(Protocol):
    async def evaluate(
        self, output: str, expected: Optional[str] = None
    ) -> Dict[str, Any]: ...


@runtime_checkable
class IObservation(Protocol):
    def trace(self, name: str): ...

    def metric(self, name: str): ...

    def log(self, level: str, msg: str): ...