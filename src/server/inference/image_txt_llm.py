import os
import sys
import asyncio
import uuid
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
from vllm.distributed.parallel_state import destroy_distributed_environment

load_dotenv()
LLM_DIR = os.getenv("LLM_IMAGE_MODEL_STORAGE")
# Disable vllm's custom logging configuration
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
# Set log level to only show critical system failures
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# Remove the (EngineCore_DP0 pid=...) prefixing
os.environ["VLLM_LOGGING_PREFIX"] = "0"
logging.getLogger("vllm").setLevel(logging.ERROR)

def get_vram_status():
    pynvml.nvmlInit()
    # Get handle for the first GPU (index 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # Convert bytes to Gigabytes
    total_gb = info.total / (1024**3)
    used_gb = info.used / (1024**3)
    free_gb = info.free / (1024**3)

    print(f"VRAM Status: {used_gb:.2f}GB / {total_gb:.2f}GB used ({free_gb:.2f}GB free)")
    pynvml.nvmlShutdown()
 
def turn_off_thinking(prompt):
    return prompt.replace(
        "\n<think>\n", 
        ""
    ).replace(
        "\n</think>\n", 
        ""
    )

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
                )
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.tokenizer = self.engine.get_tokenizer()
                self.processor = AutoProcessor.from_pretrained(
                    LLM_DIR, 
                    trust_remote_code=True,
                )
        return {
                'engine': self.engine, 
                'tokenizer': self.tokenizer,
                'processor': self.processor,
            }

    async def __del__(self):
        """Call this to free up GPU memory gracefully."""
        if self.engine:
            await self.engine.shutdown_background_loop()
            destroy_distributed_environment()
            self.engine = None

class UserSession:
    def __init__(self, 
        client_id: str,         # Application must define
        system_prompt: str,     # Application must define
        engine_data: dict,
        ):
        # Assign a unique string identifier to this specific client
        self.client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
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
        return cls(client_id, system_prompt, engine_data)

    def apply_template(self, template_obj: list):
        return self.tokenizer.apply_chat_template(
            conversation=template_obj, 
            tokenize=False, 
            add_generation_prompt=True, 
            return_tensors="pt",
            enable_thinking=False,
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
            max_tokens: int = 2048,
            verbose: bool = False):
        max_tokens = 256 if max_tokens < 256 else max_tokens

        content_in = {"role": "user"}
        
        prompt = {"role": "user"}
        if image_path:
            prompt['content'] = content_in.extend({"type": "image"})
            prompt.extend(
                {"multi_modal_data": {"image": self.decode_image(image_path)},
                    'mm_processor_kwargs': 
                        {
                            "max_pixels": max_tokens * 32 * 32, # Increase for fine detail
                            "min_pixels": 256 * 32 * 32,      # Minimum baseline
                        }
                }
            )
        else:
            prompt['content'] = prompt_str_

        # vLLM needs a unique ID for every single request
        # We combine client_id + a request hash to keep them distinct
        request_id = f"{self.client_id}-{uuid.uuid4().hex[:4]}"
        
        if not len(self.history):
            prompt_str = self.apply_template([self.system_prompt, prompt])
        else:
            prompt_str = self.apply_template([prompt])
            
        prompt_str_formatted = turn_off_thinking(prompt_str)
       
        # Appending history provides memory to the LLM, however this is the beginnning
        #  of the memory management harness.
        full_context = self.query_memory(prompt_str_formatted)

        sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=max_tokens,
            top_p = 0.95,
            top_k = 20,
        )

        if verbose:
            print(f"************\n{prompt_str_formatted}\n*****************")

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
                # pdb.set_trace()  # check completion for image data structure

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

        self.append_memory(
            self.apply_template([prompt,
                {"role": "assistant", "content": result['output']}
            ])
        )
        return result

async def main(prompt_str, image_file = None):
    if image_file:
        if not Path(image_file).is_file():
            print('Path does not address a file.')
            sys.exit(0)

    ocrAI = await UserSession.create(client_id = "Bob",
                system_prompt="You are a helpful assistant.")

    while True:        
        result = await ocrAI.inference(
            image_path=image_file, 
            prompt_str_=prompt_str,
            verbose=True,
        )
        prompt_str = input("Your reply:\n") 
        if not prompt_str:
            break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt for LLM")
    parser.add_argument("-i", "--image_path", help="Path for image to transcribe")

    args = parser.parse_args()
    asyncio.run(main(prompt_str = args.prompt, image_file = args.image_path))
