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
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from server.nodes.subagent.orchestration.llm_inference import LLMInference
import asyncio
import argparse

import asyncio

class MemoryManager:
    def __init__(self, vllm_session_id: str, system_prompt: str, context_window_length_tokens: int):
        # Keep __init__ simple and synchronous
        self.token_count = 0
        self.system_prompt = system_prompt
        self.context_window_length_tokens = context_window_length_tokens
        self.vllm_session_id = vllm_session_id
        
        # Accept the pre-configured or created LLM instance
        self.llm = LLMInference(
            client_id=vllm_session_id,
            system_prompt=system_prompt
        )

    async def create_session(self, session_id):
        result = await self.llm.create_session(self.vllm_session_id)
        return result

    async def inference(self, prompt: str, verbose: bool = False):
        if not await self.check_against_token_limit(prompt, verbose):
            return # Fail message to requester

        output = await self.llm.inference(
                client_id = self.vllm_session_id,
                prompt_str = prompt,
            )

        return output

    async def check_against_token_limit(self, string: str, verbose: bool = False):
        accept = True
        count = len(string.split(' '))
        self.token_count += count
        if self.token_count > self.context_window_length_tokens:
            accept = False
        if verbose:
            vram_status = await self.llm.get_vram_status()
            print(f"VRAM Status: {vram_status}")
        return accept
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.llm.delete_session(self.vllm_session_id)
    
async def main(
    vllm_session_id: str, 
    system_prompt: str, 
    context_window_length_tokens: int,
):
    async with MemoryManager(
            vllm_session_id = session_id,
            system_prompt = system_prompt,
            context_window_length_tokens = max_tokens, 
        ) as mm:
        
        await mm.create_session(vllm_session_id)

        result = await mm.inference(
                prompt = prompt,
                verbose = verbose,
            )

        return result['output']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-session", "--session_id", required=True, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-t", "--context_window_length_tokens", help="Max tokens for LLM to respond with", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose Output.")

    args = parser.parse_args()

    max_tokens = args.context_window_length_tokens
    system_prompt = args.system_prompt
    session_id = args.session_id
    verbose = args.verbose

    prompt = "What does Mr. Brown like to eat for breakfast?"

    output = asyncio.run(
        main(
            vllm_session_id = session_id,
            system_prompt = system_prompt,
            context_window_length_tokens = max_tokens, 
        )
    )
    print(output)
    