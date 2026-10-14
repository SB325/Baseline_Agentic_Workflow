## Description
#   This class stores "conversation" history and manages agent memory by ensuring that 
#       1. Prompt rejections by Orchestration layer are communicated to agent workflow.
#       2. Prompts and responses do not exceed the calculated KV cache allowance
#   This class shares no members between instances. 
#   S.F.B
##

# Imports
import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from orchestration.llm_orchestrator import LLM
from layers.observation.otel import otel_tracer, otel_metrics, otel_logger
import asyncio
import argparse

import asyncio

class MemoryManager:
    def __init__(self, *, orchestrator, memory_config):
        # config should be validated in the future
        self.llm = orchestrator
        self.token_count = 0
        self.observation = {'traces': otel_tracer(), 'metrics': otel_metrics(), 'logs': otel_logger()}
        
        context_config = memory_config.get('context', {}).get('context', {})
        # Default 5000 token limit per session
        self.context_window_token_limit = 5000 or \
            context_config.get('context_window_limit', None)

    async def create_session(self, session_id):
        result = await self.llm.create_session(self.llm.vllm_client_id)
        return result

    async def inference(self, prompt: str, verbose: bool = False):
        if not await self.check_against_token_limit(prompt, verbose):
            return None

        output = await self.llm.inference(
                client_id = self.llm.vllm_client_id,
                prompt_str = prompt,
            )

        return output

    async def check_against_token_limit(self, string: str, verbose: bool = False):
        self.token_count += len(string.split(' '))
        print(f"VRAM Status: {await self.llm.get_vram_status()}") if verbose else None
        return False if self.token_count > self.context_window_token_limit else True
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.llm.delete_session(self.llm.vllm_client_id)
    
async def main(
    system_prompt: str , 
    vllm_client_id: str = 'Test_ID', 
    verbose: bool = True
):
    async with MemoryManager(
        llm=LLM(client_id=vllm_client_id,
            system_prompt=system_prompt,
            )
    ) as mm:
        
        await mm.create_session(vllm_client_id)

        result = await mm.inference(
                prompt = prompt,
                verbose = verbose,
            )

        return result['output']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-session", "--session_id", required=True, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-t", "--context_window_token_limit", help="Max tokens for LLM to respond with", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose Output.")

    args = parser.parse_args()

    max_tokens = args.context_window_token_limit
    system_prompt = args.system_prompt
    session_id = args.session_id
    verbose = args.verbose

    prompt = "What does Mr. Brown like to eat for breakfast?"

    output = asyncio.run(
        main(
            vllm_client_id = session_id,
            system_prompt = system_prompt,
        )
    )
    print(output)
    