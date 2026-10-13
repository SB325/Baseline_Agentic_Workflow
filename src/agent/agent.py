# The Agentic Baseline demonstrates a multi-layered functional structure surrounding an llm
#   providing the model with 
#   context window stability (compaction), 
#   external context (mcp tools)
#   guardrails (prompt security), 
#   evaluations (quality checks) and 
#   observability (visibility)
import sys
from pathlib import Path
utils_dir = Path(__file__).resolve()
sys.path.append(str(utils_dir))
import json
import pdb
from layers.orchestration.llm_orchestrator import LLM
from layers.memory.memory_manager import MemoryManager
from layers.guardrails.policy_enforcement_manager import GuardrailManager
from layers.augmentation.augmentation_manager import AugmentationManager
from layers.evaluation.evaluation_manager import EvaluationManager
from layers.observation.observability_manager import ObservabilityManager

from load_folder_yamls import load_folder_yamls, prune_to_matching_key

class BaseAgent:
    def __init__(self, config):
        # Create all layer instances: Nested to enforce hierarchy
        agent_config = {}
        for key in config.keys():
            agent_config.update(load_folder_yamls(config.get(key, None)))

        # Semantic orchestrator, the LLM
        llm = LLM(llm_config=agent_config.get('orchestration', {}))
        # Context window manager, ensures window is not exceeded before compression
        memory = MemoryManager(orchestrator=llm, memory_config=agent_config.get('memory', {}))
        # Tool calling/ context retrieval resource
        augmentation = AugmentationManager(tool_config=agent_config.get('augmentation', {}).get('path', {}))
        # Guardrails enforce what enters/exits the context window
        guardrails = GuardrailManager(memory=memory, toolset=augmentation, guardrail_config=agent_config.get('guardrails', {})) 
        # Evaluate LLM outputs for quality
        evaluation = EvaluationManager(eval_config=agent_config.get('evaluation', {}).get('path', {}))
        # Observability traces, metrics, logs for agent
        observability = ObservabilityManager(evals=evaluation, guardrails=guardrails, obs_config=agent_config.get('observation', {}).get('path', {}))

        # Agent Harness config
        self.config = agent_config.get('harness', {}).get('path', {})

if __name__ == "__main__":
    ba = BaseAgent(
        {
            'augmentation': {
                'path': 'layers/augmentation',
                'key': 'functions'
            },
            'orchestration': {
                'path': 'layers/orchestration',
                'key': 'llm'
            },
            'memory': {
                'path': 'layers/memory',
                'key': 'context'
            },
            'evaluation': {
                'path': 'layers/evaluation',
                'key': 'test'
            }, 
            'guardrails': {
                'path': 'layers/guardrails',
                'key': 'policy'
            },
            'observation': {
                'path': 'layers/observation',
                'key': 'observation'
            },
            'agent': {
                'path': '',
                'key': 'harness'
            }
        }
    )