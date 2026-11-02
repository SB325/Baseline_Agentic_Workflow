import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import json
from dotenv import load_dotenv
import argparse
import asyncio
from util.requests_util import requests_util
from agent.layers.observation.otel import otel_tracer, otel_metrics, otel_logger
import subprocess
import ast
import re

load_dotenv()
LLM_DIR = os.getenv("LLM_IMAGE_MODEL_NT_STORAGE")
vllm_service_name = os.getenv("VLLM_SERVICE_NAME")
vllm_port = os.getenv("VLLM_PORT")

in_docker = bool(os.getenv("INDOCKER", None))

requests = requests_util(rate_limit = 1)


class LLM:
    def __init__(self, *, llm_config: dict):
        try:
            self.configs = llm_config

            self.vllm_address = self.configs.get('connection',{}).get('vllm_address', self.get_vllm_ip())
            self.client_id = self.configs.get('connection',{}).get('client_id', None)
            self.system_prompt = self.configs.get('connection',{}).get('system_prompt', None)
            self.max_tokens = self.configs.get('inference',{}).get('max_tokens',None)
            self.top_k = self.configs.get('inference',{}).get('top_k',None)
            self.top_p = self.configs.get('inference',{}).get('top_p',None)
            self.repetition_penalty = self.configs.get('inference',{}).get('repetition_penalty',None)
            self.presence_penalty = self.configs.get('inference',{}).get('presence_penalty',None)
            self.frequency_penalty = self.configs.get('inference',{}).get('frequency_penalty',None)
            self.remember = self.configs.get('inference',{}).get('remember',None)
            self.observation = {
                'traces': otel_tracer(service_name=self.client_id), 
                'metrics': otel_metrics(meter_prefix=self.client_id), 
                'logs': otel_logger(instance_id=self.client_id)
            }
            self.observation['logs'].info(f"Initiated LLM Orchestrator for ID: ({self.client_id})")
            self.observation['metrics'].create_meter(meter_name='vram', meter_type="ObservableGauge", callback_function_int=self.get_vram_status, description='VRAM Utilized.')
        except Exception as e:
            self.observation['logs'].exception(f"FAILED to Initiate LLM Orchestrator Class for ID: ({self.client_id})\n{e}")

    def get_vllm_ip(self):
        try:
            # Run a command and capture its stdout and stderr
            ip = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                    vllm_service_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            ip = ip.stdout.strip()
            return f'http://{ip}:{vllm_port}'

        except Exception as e:
            self.observation['logs'].error(f"Failed to get vLLM IP address:\n{e}")

    async def check_vllm_availability(self):
        try:
            val = requests.post(
                url_in = self.vllm_address + f'/api/status'
                )
            results = val.json()
            if not val.ok:
                self.observation['logs'].error(f"FAILED to reach vLLM Service ({self.client_id})")
                # print('VLLM Server Unavailable')
                return False
        except Exception as e:
            self.observation['logs'].error(f"Failure to assess vLLM availability.\n{e}")
        return True

    async def get_vram_status(self):
        try:
            status = False
            if not self.check_vllm_availability:
                return {'status': status, 'output': "VLLM server Unavailable!"}

            val = requests.get(
                url_in = self.vllm_address + f'/api/get_gpu_status',
                )
            results = val.json()
            
            if val.ok:
                status = True
            match = re.search(r"\(([\d.]+)GB free\)", results['data']['result'])
            val_out = 0
            if match:
                val_out = float(match.group(1))

            # return json.dumps({'status': status, 'output': results['data']['result']})
            return val_out

        except Exception as e:
            self.observation['logs'].error(f"Failed to get vram_status:\n{e}")

    async def create_session(self,
            client_id: str = None,
            system_prompt: str = None,
        ):
        try:
            status = False
            if not self.check_vllm_availability:
                return {'status': status, 'output': "VLLM server Unavailable!"}

            assert client_id or self.client_id, "There must be a client_id"
            client_id = client_id or self.client_id
            assert system_prompt or self.system_prompt, "There must be a system prompt"
            system_prompt = system_prompt or self.system_prompt

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
            
            if not status:
                return {'status': ast.literal_eval(results['detail'])['status_code'], 'output': ast.literal_eval(results['detail'])['detail']}

            return {'status': results['status'], 'output': results['message']}
        except Exception as e:
            self.observation['logs'].error(f"Failed to create session -{client_id}-:\n{e}")

    async def delete_session(self,
            client_id: str = None,
        ):
        try:
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
            
            detail = None
            if results.get('detail', None):
                detail = ast.literal_eval(results['detail'])
                if detail.get('status_code', None):
                    status = detail['status_code']
                    detail = detail['detail']
            
            if not status:
                return {'status': status, 'output': detail}
            return {'status': status}
        except Exception as e:
            self.observation['logs'].error(e)

    async def inference(self, 
            *,
            client_id,
            prompt_string,
            max_tokens=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            remember=None
        ):
        try:
            # If values are presented as arguments to the inference call, those take precidence
            # over yaml config values.
            max_tokens = max_tokens or self.max_tokens
            top_p = top_p or self.top_p
            top_k = top_k or self.top_k 
            repetition_penalty = repetition_penalty or self.repetition_penalty
            presence_penalty = presence_penalty or self.presence_penalty
            frequency_penalty = frequency_penalty or self.frequency_penalty
            remember = remember or self.remember

            status = False
            if not self.check_vllm_availability:
                return {'status': status, 'output': "VLLM server Unavailable!"}

            assert client_id, "There must be a client_id"
            assert prompt_string, "There must be a prompt_str"

            with self.observation['traces'].set_span('orchestrator.inference'):
                self.observation['logs'].info(f'Prompt: {prompt_string}')
                val = requests.post(
                    url_in = self.vllm_address + f'/api/inference_on_session',
                    json_in = {
                        'client_id': client_id, 
                        'prompt_string': prompt_string,
                        'max_tokens': max_tokens,
                        'top_p': top_p,
                        'top_k': top_k,
                        'repetition_penalty': repetition_penalty,
                        'presence_penalty': presence_penalty,
                        'frequency_penalty': frequency_penalty,
                        'remember': remember 
                    }
                )
                
                results = val.json()
                status_code = None
                detail = None
                if results.get('detail', None):
                    detail = ast.literal_eval(results['detail'])
                    if detail.get('status_code', None):
                        status_code = detail['status_code']
                        self.observation['logs'].info(f'Inference completed with status {status_code}')    
                        detail = detail['detail']
                        self.observation['logs'].info(f'Detail: {detail}')

                if not val.ok:
                    self.observation['logs'].info(
                        f"Inference POST request failed: {val.status_code}: {val.reason}")
                    return {'status': status_code, 'output': results['detail']}
                
                self.observation['logs'].info(
                    f"Output: {results['data']['result']['output']} or 'Empty'")
                return {'status': results['status'], 'output': results['data']['result']['output']}
        except Exception as e:
            self.observation['logs'].error(f"Inference failed:\n{e}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-client", "--client_id", required=False, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-t", "--max_tokens", help="Max tokens for LLM to respond with", type=int, default=8000)
    parser.add_argument("-c", "--conversate", action='store_true', help="Conversational interraction (remember history)")
    parser.add_argument("-api", "--use_api", action='store_true', help="Run inference through running vLLM API.")

    args = parser.parse_args()

    max_tokens = args.max_tokens
    conversate = args.conversate
    system_prompt = args.system_prompt
    use_api = args.use_api
    client_id = args.client_id

    llm = LLM(llm_config=
        {   'connection': {
                'client_id': client_id or "Test_ID",
                'system_prompt': 'You are a helpful Assistant.'
            }
        }
    )
    # pdb.set_trace()
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
                client_id = client_id,
                prompt_string = prompt
            )
        )
        print(output)
    asyncio.run(llm.delete_session(client_id))
        
