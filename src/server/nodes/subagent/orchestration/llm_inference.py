import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import json
from dotenv import load_dotenv
import argparse
import asyncio
from util.requests_util import requests_util
import subprocess

load_dotenv()
LLM_DIR = os.getenv("LLM_IMAGE_MODEL_NT_STORAGE")
vllm_service_name = os.getenv("VLLM_SERVICE_NAME")
vllm_port = os.getenv("VLLM_PORT")

in_docker = bool(os.getenv("INDOCKER", None))

requests = requests_util(rate_limit = 1)

def get_vllm_ip():
    if bool(in_docker):
        return f'{vllm_service_name}:{vllm_port}'

    # Run a command and capture its stdout and stderr
    ip = subprocess.run(
        f"docker inspect --format='{{{{.NetworkSettings.Networks.homeserver.IPAddress}}}}' {vllm_service_name}",
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode output as text (UTF-8 by default)
        shell=True           # Raise CalledProcessError if the command returns a non-zero exit code
    ).stdout.replace('\n', '')

    return f'http://{ip}:{vllm_port}'


class LLMInference:
    def __init__(self, 
            vllm_address: str = get_vllm_ip(),
            system_prompt: str = "You are a helpful assistant.",
            client_id: str = None,  
        ):
        self.vllm_address = vllm_address
        self.client_id = client_id
        self.system_prompt = system_prompt

    async def check_vllm_availability(self):
        try:
            val = requests.post(
                url_in = self.vllm_address + f'/api/status'
                )
            results = val.json()
            if not val.ok:
                print('VLLM Server Unavailable')
                return False
        except:
            return False
        return True

    async def get_vram_status(self):
        status = False
        if not self.check_vllm_availability:
            return {'status': status, 'output': "VLLM server Unavailable!"}

        val = requests.get(
            url_in = self.vllm_address + f'/api/get_gpu_status',
            )
        results = val.json()

        if val.ok:
            status = True
        
        return json.dumps({'status': status, 'output': results['data']['result']})

    async def create_session(self,
            client_id: str = None,
            system_prompt: str = None,
        ):
        status = False
        if not self.check_vllm_availability:
            return {'status': status, 'output': "VLLM server Unavailable!"}

        if not self.client_id:
            assert client_id, "There must be a client_id"
        else:
            client_id = self.client_id
        if not self.system_prompt:
            assert system_prompt, "There must be a system prompt"
        else:
            system_prompt = self.system_prompt

        val = requests.post(
            url_in = self.vllm_address + f'/api/create_session',
            json_in = {
                'client_id': client_id, 
                'system_prompt': system_prompt,
                }
            )
        results = val.json()
        
        if val.ok:
            status = True
        
        return {'status': status, 'output': results['message']}

    async def delete_session(self,
            client_id: str = None,
        ):
        status = False
        if not self.check_vllm_availability:
            return {'status': status, 'output': "VLLM server Unavailable!"}

        assert client_id, "There must be a client_id"

        val = requests.post(
            url_in = self.vllm_address + f'/api/delete_session',
            json_in = {
                'client_id': client_id, 
                }
            )
        results = val.json()

        if val.ok:
            status = True
        
        return results['status']

    async def inference(self, 
            client_id: str = None,
            prompt_str: str = None,
            max_tokens: int = 256,
        ):
        status = False
        if not self.check_vllm_availability:
            return {'status': status, 'output': "VLLM server Unavailable!"}

        assert client_id, "There must be a client_id"
        assert prompt_str, "There must be a prompt_str"

        val = requests.post(
            url_in = self.vllm_address + f'/api/inference_on_session',
            json_in = {
                'client_id': client_id, 
                'prompt_str': prompt_str,
                'max_tokens': max_tokens,
                }
            )
        results = val.json()

        if val.ok:
            status = True
        return results['data']['result']['output']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-session", "--session_id", required=True, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-t", "--max_tokens", help="Max tokens for LLM to respond with", type=int, default=8000)
    parser.add_argument("-c", "--conversate", action='store_true', help="Conversational interraction (remember history)")
    parser.add_argument("-api", "--use_api", action='store_true', help="Run inference through running vLLM API.")

    args = parser.parse_args()

    max_tokens = args.max_tokens
    conversate = args.conversate
    system_prompt = args.system_prompt
    use_api = args.use_api
    client_id = args.session_id

    llm = LLMInference(
            client_id = client_id,
            system_prompt = system_prompt
        )
    
    print(asyncio.run(llm.get_vram_status()))
    print(asyncio.run(llm.create_session()))

    animals = [
        "animal with four legs?", 
        "animal that can fly?", 
        "animal that has scales for skin?", 
        "animal that is very tall?"
    ]
    cnt = 0

    while True:
        cnt += 1
        if cnt > len(animals):
            break
          
        prompt = f"What is the name of an {animals[cnt-1]}"
        print(prompt)
        output = asyncio.run(
            llm.inference(
                prompt_str = prompt
            )
        )
        print(output)

    asyncio.run(llm.delete_session(client_id))
        
