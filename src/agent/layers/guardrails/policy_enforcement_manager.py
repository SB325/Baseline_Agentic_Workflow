## Description
#   This class facilitates agent policy enforcement to ensure that 
#   actions and outputs are safe from spillage, prompt injection, 
#   improper language.
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
from memory.memory_manager import MemoryManager
from augmentation.augmentation_manager import AugmentationManager
from load_folder_yamls import load_folder_yamls, prune_to_matching_key

class GuardrailManager():
    def __init__(self, *, memory, toolset, guardrail_config):
        self.config = guardrail_config.get('policy', {}).get('policy_config_vals', None)
        self.memory = memory
        self.augmentation = toolset
        self.failure_flag = False

    def vulgarity_check(self, text: str):
        # set failure flag if test failed
        if self.failure_flag:
            return True
    
    def prompt_injection_check(self, text: str):
        # set failure flag if test failed
        if self.failure_flag:
            return True

    def spillage_check(self, text: str):
        # set failure flag if test failed
        if self.failure_flag:
            return True

    def check_all(self, context: str):
        check = f'Guardrail check: \n'
        check = check + 'Vulgarity Check: '
        check = check + 'FAILED\n' if self.vulgarity_check(context) else 'PASSED\n'
        check = check + 'Prompt Inj Check: '
        check = check + 'FAILED\n' if self.prompt_injection_check(context) else 'PASSED\n'
        check = check + 'Spillage Check: '
        check = check + 'FAILED\n' if self.spillage_check(context) else 'PASSED\n'
        return self.failure_flag

async def main(
    system_prompt: str = 'You are a helpful Assistant.', 
    vllm_client_id: str = 'Test_ID', 
    prompt: str = 'What time is it right now?',
    verbose: bool = True
):
    async with GuardrailManager(
        memory=MemoryManager(
            llm=LLM(client_id=vllm_client_id,
                system_prompt=system_prompt,
                )
        ),
        toolset=AugmentationManager(
            load_folder_yamls({'guardrails': {
                    'path': 'layers/guardrails',
                    'key': 'policy'
                }
            })
        ),
        guardrail_config={} # Will be required
    ) as gr:
        
        await gr.memory.create_session(vllm_client_id)

        result = await gr.memory.inference(
                prompt = prompt,
                verbose = verbose,
            )
        if not gr.check_all(result['output']):
            print(f"LLM's returned context violated policy")
            # Reset failure flag
            self.failure_flag = False
        return result['output']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-client", "--client_id", required=False, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-p", "--prompt", help="Prompt for the LLM")
    args = parser.parse_args()

    system_prompt = args.system_prompt
    client_id = args.client_id
    prompt = args.prompt

    output = asyncio.run(
        main(
            vllm_client_id = client_id,
            system_prompt = system_prompt,
            prompt = prompt,
        )
    )
    print(output)
    