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
            self.observation = {'traces': otel_tracer(self.client_id), 'metrics': otel_metrics(self.client_id), 'logs': otel_logger(self.client_id)}
            self.observation['logs'].info(f"Initiated LLM Orchestrator for ID: ({self.client_id})")
            self.observation['metrics'].create_meter(meter_name='vram', meter_type="ObservableGauge", callback=self.get_vram_status, description='VRAM Utilized.')
            pdb.set_trace()
        except:
            self.observation['logs'].exception(f"FAILED to Initiate LLM Orchestrator Class for ID: ({self.client_id})")

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
            self.observation['logs'].error(e)

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
            self.observation['logs'].error(e)
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

            return json.dumps({'status': status, 'output': results['data']['result']})
        except Exception as e:
            self.observation['logs'].error(e)

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
            self.observation['logs'].error(e)

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
            prompt_str,
            max_tokens,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            remember
        ):
        try:
            # If values are presented as arguments to the inference call, those take precidence
            # over yaml config values.
            max_tokens = max_tokens or self.max_tokens
            top_p = top_p or self.top_p
            top_k = self.top_k or top_k
            repetition_penalty = repetition_penalty or self.repetition_penalty
            presence_penalty = presence_penalty or self.presence_penalty
            frequency_penalty = requency_penalty or self.frequency_penalty
            remember = remember or self.remember

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
                    'top_p': top_p,
                    'top_k': top_k,
                    'repetition_penalty': repetition_penalty,
                    'presence_penalty': presence_penalty,
                    'frequency_penalty': frequency_penalty,
                    'remember': remember 
                }
            )
            results = val.json()
            
            detail = None
            if results.get('detail', None):
                detail = ast.literal_eval(results['detail'])
                if detail.get('status_code', None):
                    status_code = detail['status_code']
                    detail = detail['detail']

            if not val.ok:
                return {'status': status_code, 'output': results['detail']}

            return {'status': results['status'], 'output': results['data']['result']['output']}
        except Exception as e:
            self.observation['logs'].error(e)
            
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
                prompt_str = prompt
            )
        )
        print(output)

    asyncio.run(llm.delete_session(client_id))
        
