## Description
#   This class manages agent memory by ensuring that 
#       1. Prompt rejections by Orchestration layer are communicated to agent workflow.
#       2. Prompts and responses do not exceed the calculated KV cache allowance
#   This class shares no members between instances. 
##

# Imports
import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from orchestration.llminference import LLMInference
import asyncio

class MemoryManager:
    def __init__(self, 
            vllm_session_id: str,
            system_prompt: str,
            prompt: str, 
            context_window_length_tokens: int, 
        ):
        self.token_count = 0
        self.system_prompt = system_prompt

        if not check_against_token_limit(system_prompt):
            return # Fail message to requester

        llm = LLMInference(
            client_id = vllm_session_id,
            system_prompt = system_prompt
        )
        if not prompt:
            return 

        self.prompt = prompt
        if not check_against_token_limit(prompt):
            return # Fail message to requester

        output = asyncio.run(
            llm.inference(
                prompt_str = prompt
            )
        )

        return output
    
    def inference(prompt: str):
        if not check_against_token_limit(prompt):
            return # Fail message to requester

        output = asyncio.run(
            llm.inference(
                prompt_str = prompt
            )
        )

        return output

    def check_against_token_limit(string: str):
        accept = True
        count = len(string.split(' '))
        self.token_count += count
        if self.token_count > context_window_length_tokens:
            accept = False
        
        return accept
