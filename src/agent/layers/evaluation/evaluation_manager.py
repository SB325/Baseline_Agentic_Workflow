## Description
#   This class facilitates agent evaluation to ensure quality of 
#   agent outputs in terms of format, completeness, and accuracy.
#   This class shares no members between instances. 
#   S.F.B
##

# Imports
import os, sys
import pdb
import json
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from orchestration.llm_orchestrator import LLM
from memory.memory_manager import MemoryManager
from augmentation.augmentation_manager import AugmentationManager
from load_folder_yamls import load_folder_yamls, prune_to_matching_key

class EvaluationManager():
    def __init__(self, *, eval_config):
        self.config = eval_config
    
    def format_eval(self, fmt: str):
        if 'json' == fmt:
            try: 
                test_json = json.loads(fmt)
            except:
                print('Not Valid Json!')
                return {'Passed': False, 'Test': 'JSON Formatting'}
        else:
            pass

    def classifier_eval(self, question: str, options: list):
        options = options if 'other' in options else options.append('other')
        # Try with laya running as a locally hosted service
        pass

def main(config):
    em = EvaluationManager(eval_config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-client", "--client_id", required=False, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-p", "--prompt", help="Prompt for the LLM")
    args = parser.parse_args()

    output = asyncio.run(
        main(
            load_folder_yamls({
                'path': 'layers/evaluation',
                'key': 'test'
            })
        )
    )
    print(output)
    