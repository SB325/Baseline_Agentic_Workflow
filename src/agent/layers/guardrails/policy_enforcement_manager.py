"""
GuardrailManager — policy enforcement for agent inputs and outputs.
Depends on IAugmentation only; no nested memory class, no OpenTelemetry.
"""
from typing import Any, Dict
from agent.interfaces import IGuardrail, IAugmentation


class GuardrailManager:
    def __init__(
        self,
        *,
        toolset: IAugmentation,
        guardrail_config: dict,
    ):
        self.augmentation = toolset
        self.config = guardrail_config.get("policy", {}).get("policy_config_vals", None)
        self.failure_flag = False

    async def check_input(self, text: str, context: Dict[str, Any]) -> bool:
        # TODO: integrate real NSFW / prompt-injection / spillage models
        return self.failure_flag

    async def check_output(self, text: str, context: Dict[str, Any]) -> bool:
        # TODO: integrate real NSFW / prompt-injection / spillage models
        return self.failure_flag
