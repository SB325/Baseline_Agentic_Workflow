import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Disable vllm's custom logging configuration
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
# Set log level to only show critical system failures
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# Remove the (EngineCore_DP0 pid=...) prefixing
os.environ["VLLM_LOGGING_PREFIX"] = "0"

import signal
import asyncio
import uuid
import json
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoProcessor
import pdb
import logging
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import pynvml
import base64
from io import BytesIO
import argparse
from pathlib import Path
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
import torch.distributed as dist
from util.requests_util import requests_util
import builtins
builtins.PILImageResampling = Image.Resampling

load_dotenv()
LLM_DIR = os.getenv("LLM_IMAGE_MODEL_NT_STORAGE")
vllm_service_name = os.getenv("VLLM_SERVICE_NAME")
indocker = bool(os.getenv("INDOCKER"))
logging.getLogger("vllm").setLevel(logging.ERROR)

requests = requests_util(rate_limit = 1)

def get_vram_status(verbose: bool = False):
    pynvml.nvmlInit()
    # Get handle for the first GPU (index 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # Convert bytes to Gigabytes
    total_gb = info.total / (1024**3)
    used_gb = info.used / (1024**3)
    free_gb = info.free / (1024**3)

    msg = f"\nVRAM Status: {used_gb:.2f}GB / {total_gb:.2f}GB used ({free_gb:.2f}GB free)\n"
    if verbose:
        print(msg)

    pynvml.nvmlShutdown()

    return msg
 
def turn_off_thinking(prompt):
    template = prompt.replace(
        "\n<think>\n\n</think>\n\n", 
        ""
    ).replace(
        "<|im_end|>",
        "/no_think<|im_end|>", 
    )
        
    return template

def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

class InferenceEngine:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance.engine = None
        return cls._instance

    async def get_engine(self):
        async with self._lock:
            if self.engine is None:
                engine_args = AsyncEngineArgs(
                    model=LLM_DIR,
                    gpu_memory_utilization=0.8,
                    trust_remote_code=True,
                    reasoning_parser=None,
                    max_model_len=8096,
                )
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.tokenizer = self.engine.get_tokenizer()
                self.processor = AutoProcessor.from_pretrained(
                    LLM_DIR, 
                    trust_remote_code=True,
                    use_fast=False
                )
        return {
                'engine': self.engine, 
                'tokenizer': self.tokenizer,
                'processor': self.processor,
            }

    async def __del__(self):
        """Call this to free up GPU memory gracefully."""
        if self.engine:
            if hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            # Check for V0 engine (AsyncLLMEngine)
            elif hasattr(self.engine, "shutdown_background_loop"):
                self.engine.shutdown_background_loop()

            # destroy_distributed_environment()
            destroy_model_parallel()
            print("vLLM Engine shut down gracefully.")
            
            self.engine = None
            if dist.is_initialized():
                dist.destroy_process_group()

class UserSession:
    def __init__(self, 
        client_id: str,         # Application must define
        engine_data: dict,
        system_prompt: str = "You are a helpful assistant.",
        ):
        # Assign a unique string identifier to this specific client
        self.client_id = f"client-{client_id}-{uuid.uuid4().hex[:8]}"
        self.shared_engine = InferenceEngine()
        self.history = []
        self.engine = engine_data['engine']
        self.tokenizer = engine_data['tokenizer']
        self.processor = engine_data['processor']
        self.system_prompt = {
            "role": "system", 
            "content": system_prompt,
        }

    @classmethod
    async def create(cls, client_id: str, system_prompt: str):
        shared_instance = InferenceEngine()
        # This safely awaits the singleton's internal lock and initialization
        engine_data = await shared_instance.get_engine()
        
        # Return a fully initialized instance of UserSession
        return cls(client_id = client_id, 
            system_prompt=system_prompt, 
            engine_data=engine_data)

    def apply_template(self, template_obj: list, tools: list =None):
        return self.tokenizer.apply_chat_template(
            conversation=template_obj, 
            tokenize=False, 
            add_generation_prompt=True, 
            return_tensors="pt",
            enable_thinking=False,
            tools=tools,
            # continue_final_message=True, #  The model will continue this message rather than starting a new one
            # tools=[],  Each tool should be passed as a JSON Schema, giving the name, description and argument types for the tool.
        )

    def decode_image(self, image_file: str):
        extension = os.path.splitext(image_file)[1].lower()
        standard_img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

        if 'pdf' in extension:
            pages = convert_from_path(image_file, dpi=300)
            raw_img = pages[0].convert("RGB")
            # raw_img.save("output_image.png")
            return raw_img #pil_to_base64(raw_img)
        elif extension in standard_img_extensions:
            raw_img = Image.open(image_file)
            # raw_img.save("output_image.png")
            return raw_img #pil_to_base64(raw_img.convert("RGB"))
        else:
            raise ValueError(f"Extension {extension} is not recognized.")

    def query_memory(self, templated_prompt: str):
        """
        Input to memory harness, will make this an abstract method in the future
        """
        self.history.append(templated_prompt)
        return ''.join(self.history)

    def append_memory(self, templated_response: str):
        """
        Input to memory harness, will make this an abstract method in the future
        """
        self.history.append(templated_response)

    async def inference(self, 
            image_path: str,
            prompt_str_: str,
            max_tokens: int = 256,
            remember: bool = False,
            verbose: bool = False):
        max_tokens = 64 if max_tokens < 64 else max_tokens
        print(f"Max Tokens: {max_tokens}")
        
        prompt = {"role": "user"}
        if image_path:
            prompt['content'] = [
                {
                    "type": "image", 
                    "image": self.decode_image(image_path),
                    "max_pixels": max_tokens * 32 * 32, # Increase for fine detail
                    "min_pixels": 256 * 32 * 32,      # Minimum baseline
                },
                {
                    "type": "text", 
                    "text": prompt_str_,
                },
            ]
        else:
            prompt['content'] = prompt_str_

        # tools = [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "get_weather",
        #             "description": "Get current weather for a city",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {"type": "string", "description": "City name"}
        #                 },
        #                 "required": ["location"]
        #             }
        #         }
        #     }
        # ]

        # vLLM needs a unique ID for every single request
        # We combine client_id + a request hash to keep them distinct
        request_id = f"request-client-{self.client_id}-{uuid.uuid4().hex[:4]}"
        
        if not len(self.history):
            prompt_str = self.apply_template([ self.system_prompt, prompt ])
        else:
            prompt_str = self.apply_template([prompt])
            
        # prompt_str = turn_off_thinking(prompt_str)

        # Appending history provides memory to the LLM, however this is the beginnning
        #  of the memory management harness.
        full_context = self.query_memory(prompt_str)
        sampling_params = SamplingParams(
            temperature=0.8, 
            max_tokens=max_tokens,
            top_p = 0.9,
            top_k = 20,
            repetition_penalty=1.1,  # Value > 1.0: Discourages repetition
            presence_penalty=0.1,  # Value > 0.0: Encourages the model to talk about new topics.
            frequency_penalty=0.1,  # Value > 0.0: Increases the penalty each time a token is repeated
        )

        if verbose:
            print(f"************\n{prompt_str}\n*****************")

        results_generator = self.engine.generate(
            prompt=full_context, 
            sampling_params=sampling_params, 
            request_id=request_id,
        )
    
        reason = None
        result = {'status': None, 'output': None}
        async for request_output in results_generator:
            for completion in request_output.outputs:
                reason = completion.finish_reason

        result['output'] = completion.text   
        if reason == "length":
            result['status'] = ("The response was cut off (max_tokens).")
        elif reason == "stop":
            result['status'] = ("The model finished naturally.")
        elif reason == "abort":
            result['status'] = ("The request was aborted.")
        if verbose:
            print(f"Status:\n{result['status']}")
            print(f"Generated text:\n{result['output']}")

        # append response to context memory
        if remember:
            self.append_memory(
                self.apply_template([prompt,
                    {"role": "assistant", "content": result['output']}
                ])
            )
        return result

async def startup_engine():
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    async def shutdown(sig_name):
        print(f"\nReceived {sig_name}. Cleaning up...")
        # Access the engine through your ocrAI session
        await ocrAI.shared_engine.__del__() 
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, 
            lambda s=sig: asyncio.create_task(shutdown(s.name))
        )

    new_instance = InferenceEngine()
    _ = await new_instance.get_engine()

async def main(
        prompt_str, 
        max_tokens, 
        image_file = None, 
        conversate = False,
        system_prompt = "You are a concise assistant. Provide direct, information-dense answers only.",
        client_id = "",
        use_api = False,
    ):
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()


    async def shutdown(sig_name):
        print(f"\nReceived {sig_name}. Cleaning up...")
        # Access the engine through your ocrAI session
        await ocrAI.shared_engine.__del__() 
        stop_event.set()
        
    ip = ""
    if use_api:
        
        if not indocker:
            ip = "localhost"
        else:
            import subprocess
            # Use list format without shell=True to avoid shell interpretation issues
            # The format string with curly braces is passed directly to docker
            result = subprocess.run(
                [
                    "docker", "inspect",
                    "--format={{.NetworkSettings.Networks.homeserver.IPAddress}}",
                    vllm_service_name
                ],
                capture_output=True,
                text=True
            )
            ip = result.stdout.strip()
            
        endpoint = f'http://{ip}:8000'

    if image_file:
        if not Path(image_file).is_file():
            print('Path does not address a file.')
            sys.exit(0)

    if use_api:
        response = requests.post(
            url_in = f"{endpoint}/api/create_session", 
            data_in = json.dumps({
                    'client_id': client_id,
                    'system_prompt': system_prompt
            }), 
            headers_in = {'Content-Type': 'application/json'}
        )
        if not response.ok:
            print(f"Failed to create chat session: [{response.status_code}] - {response.json()}")
            return None
    else:
        ocrAI = await UserSession.create(
                client_id = client_id,
                system_prompt= system_prompt,
            )
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, 
            lambda s=sig: asyncio.create_task(shutdown(s.name))
        )

    while True:  
        if use_api:
            response = requests.post(
                url_in = f"{endpoint}/api/inference_on_session", 
                data_in = json.dumps({
                    'client_id': client_id,
                    'prompt_str': prompt_str,
                    'image_file': image_file,
                    'max_tokens': max_tokens,
                }),
                headers_in = {'Content-Type': 'application/json'}
            )
            
            if not response.ok:
                print(f"Failed to perform inference for chat session {client_id}: [{response.status_code}] - {response.json}")
                return None
            result = response['data']['result']
            print(result)
        else:      
            result = await ocrAI.inference(
                image_path=image_file, 
                prompt_str_=prompt_str,
                max_tokens=int(max_tokens),
                remember=conversate,
                verbose=False,
            )
            prompt_str = input("Your reply: (or press Enter to quit):\n") 
            if not prompt_str:
                await shutdown("Manual Exit")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt for LLM")
    parser.add_argument("-session", "--session_id", required=True, help="Session Identifier.")
    parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    parser.add_argument("-i", "--image_path", help="Path for image to transcribe")
    parser.add_argument("-t", "--max_tokens", help="Max tokens for LLM to respond with", type=int, default=8000)
    parser.add_argument("-c", "--conversate", action='store_true', help="Conversational interraction (remember history)")
    parser.add_argument("-api", "--use_api", action='store_true', help="Run inference through running vLLM API.")

    args = parser.parse_args()
    asyncio.run(
        main(
            prompt_str = args.prompt, 
            image_file = args.image_path, 
            max_tokens = args.max_tokens,
            conversate = args.conversate,
            system_prompt = args.system_prompt,
            use_api = args.use_api,
            client_id = args.session_id,
        )
    )
