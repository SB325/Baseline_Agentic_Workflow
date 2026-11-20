"""
Dependency injection wiring for the Agentic Baseline.
Instantiates concrete layer classes and injects them into BaseAgent.
"""
from agent.interfaces import (
    ILLM,
    IMemory,
    IGuardrail,
    IAugmentation,
    IEvaluation,
    IObservation,
)
from agent.agent import BaseAgent
from agent.layers.orchestration.llm_orchestrator import LLM
from agent.layers.memory.memory_manager import MemoryManager
from agent.layers.guardrails.policy_enforcement_manager import GuardrailManager
from agent.layers.augmentation.augmentation_manager import AugmentationManager
from agent.layers.evaluation.evaluation_manager import EvaluationManager
from agent.layers.observation.otel import OTelObservation
import pdb

def build_agent_from_config(config: dict) -> BaseAgent:
    observation: IObservation = OTelObservation(service_name="agentic-baseline")

    llm: ILLM = LLM(llm_config=config.get("orchestration", {}))
    memory: IMemory = MemoryManager(memory_config=config.get("memory", {}))
    # pdb.set_trace()
    augmentation: IAugmentation = AugmentationManager(
        tool_config=config.get("augmentation", {})
    )
    guardrails: IGuardrail = GuardrailManager(
        toolset=augmentation,
        guardrail_config=config.get("guardrails", {}),
    )
    evaluation: IEvaluation = EvaluationManager(
        eval_config=config.get("evaluation", {})
    )

    return BaseAgent(
        llm=llm,
        memory=memory,
        guardrails=guardrails,
        augmentation=augmentation,
        evaluation=evaluation,
        observation=observation,
    )