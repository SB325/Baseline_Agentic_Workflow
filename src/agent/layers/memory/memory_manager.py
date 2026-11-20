"""
MemoryManager — context window token accounting.
Depends only on IMemory protocol; no LLM coupling, no OpenTelemetry.
"""
from agent.interfaces import IMemory


class MemoryManager:
    def __init__(self, memory_config: dict = {}):
        self.token_count = 0
        context_config = memory_config.get("context", {}).get("context", {})
        # Default 5000 token limit per session
        self.context_window_token_limit = 5000 or \
            context_config.get("context_window_limit", None)

    async def check_against_token_limit(
        self, string: str, verbose: bool = False
    ) -> bool:
        self.token_count += len(string.split(' '))
        if verbose:
            print(f"Token count: {self.token_count} / {self.context_window_token_limit}")
        return False if self.token_count > self.context_window_token_limit else True
