import os
from dotenv import load_dotenv
import logging
load_dotenv()
LLM_DIR = os.getenv("LLM_MODEL_STORAGE")
# Disable vllm's custom logging configuration
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
# Set log level to only show critical system failures
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# Remove the (EngineCore_DP0 pid=...) prefixing
os.environ["VLLM_LOGGING_PREFIX"] = "0"
logging.getLogger("vllm").setLevel(logging.ERROR)

import uuid
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import pynvml
import pdb
import re
from datetime import datetime

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

class VLLMSingleton:
    _instance = None
    @classmethod
    def get_engine(cls):
        if cls._instance is None:
            engine_args = AsyncEngineArgs(
                model=LLM_DIR,
                dtype="float16",
                enable_prefix_caching=True, # Critical for fast history
                gpu_memory_utilization=0.90,
                max_model_len=8192,             # Context window size
                max_num_batched_tokens=8192,    # Concurrently processed tokens
            )
            cls._instance = AsyncLLMEngine.from_engine_args(engine_args)
        return cls._instance

    @classmethod
    async def shutdown(cls):
        """Cleanly stop the engine and background workers."""
        if cls._instance is not None:
            print("Shutting down vLLM engine...")
            await asyncio.sleep(1.0) 
            cls._instance = None
            print("Engine shut down successfully.")

class UserSession:
    def __init__(self, 
            user_id: str, 
            temp_setting=0.7,
            system_prompt: str = "You are a helpful assistant."):
        self.user_id = user_id
        self.temp_setting = temp_setting
        self.engine = VLLMSingleton.get_engine()
        self.current_request_id = None  # Track the active task
        # Initialize history with the system prompt
        self.history = [{"role": "system", "content": system_prompt}]
        self.max_model_len = self.engine.vllm_config.model_config.max_model_len
    
    def _format_chat(self, thinking_mode_on: bool) -> str:
        """
        Manually formats history into Qwen's ChatML format.
        <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant
        """
        prompt = ""
        for msg in self.history:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            if not thinking_mode_on:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>/nothink\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    async def _save_thought_to_disk(self, thought_text: str):
        """Saves the <think> block to a local file."""
        log_dir = "logs/thoughts"
        os.makedirs(log_dir, exist_ok=True)
        
        filename = f"{log_dir}/{self.user_id}_thoughts.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"--- {timestamp} ---\n{thought_text.strip()}\n\n")

    async def generate(self, user_input: str,
            add_thinking_output: bool = False,
            max_tokens: int = 512,
            fill_fraction_limit=0.5):

        fill_fraction = (len(user_input)+max_tokens)/(self.max_model_len)
        if fill_fraction > fill_fraction_limit:
            return f"[ERROR]: Prompt and max_tokens response will take up " + \
                f"{fill_fraction*100:.1f}% of context window, " + \
                 f"Need at least {(1-fill_fraction_limit)*100}% of the " + \
                 "window for conversation history."
        
        user_input = {"role": "user", "content": user_input}
        self.history.append(user_input)
        
        full_prompt = self._format_chat(add_thinking_output)

        self.current_request_id = f"{self.user_id}-{uuid.uuid4()}"

        sampling_params = SamplingParams(
            temperature=self.temp_setting, # randomness
            max_tokens=max_tokens,         # max tokens to return
            top_p=0.95,             # avoids long tail tokens, probability threshold
            top_k=20,               # k most likely next words to consider
            presence_penalty=1.5,   # avoids getting stuck in loops
            repetition_penalty=1.1, # likeliness for repeated words
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        # 3. Stream from the engine
        results_generator = self.engine.generate(full_prompt, sampling_params, self.current_request_id)
        
        raw_text = ""
        async for request_output in results_generator:
            raw_text = request_output.outputs[0].text
        
        response_text = raw_text
        if add_thinking_output:
            # Regex to find everything between <think> tags
            thought_match = re.search(r'<think>(.*?)</think>', raw_text, flags=re.DOTALL)
            if thought_match:
                thought_content = thought_match.group(1)
                await self._save_thought_to_disk(raw_text)
        else:
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            response_text = clean_text
        
        self.history.append({"role": "assistant", "content": response_text})

        return response_text
    
    def get_vram_status(self):
        get_vram_status()

    async def abort(self):
        """Immediately stops the current generation on the GPU."""
        if self.current_request_id:
            print(f"Aborting request: {self.current_request_id}")
            # vLLM's AsyncLLMEngine.abort is a sync method in most versions
            # but wrapping in a check for safety
            self.engine.abort(self.current_request_id)
            self.current_request_id = None

# --- Example Usage ---
async def main():
    try:
        user = "Alice"
        alice = UserSession(user)
        
        # First turn
        msg ="My name is Alice. Remember that."
        print(f"{user}: {msg}")
        resp1 = await alice.generate(msg)
        print(f"Bot: {resp1}")
        
        # Second turn (The model will remember the name)
        msg = "What is my name?"
        print(f"{user}: {msg}")
        resp2 = await alice.generate(msg)
        print(f"Bot: {resp2}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # CRITICAL: This prevents the 'died unexpectedly' error
        await VLLMSingleton.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
