import uuid
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import os
from dotenv import load_dotenv
import pynvml
import pdb

load_dotenv()
LLM_DIR = os.getenv("LLM_MODEL_STORAGE")

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
                trust_remote_code=True,
                quantization="awq", 
                enable_prefix_caching=True, # Critical for fast history
                gpu_memory_utilization=0.80
            )
            cls._instance = AsyncLLMEngine.from_engine_args(engine_args)
        return cls._instance

class UserSession:
    def __init__(self, 
            user_id: str, 
            temp_setting=0.7,
            system_prompt: str = "You are a helpful assistant."):
        self.user_id = user_id
        self.temp_setting = temp_setting
        self.engine = VLLMSingleton.get_engine()
        # Initialize history with the system prompt
        self.history = [{"role": "system", "content": system_prompt}]

    def _format_chat(self) -> str:
        """
        Manually formats history into Qwen's ChatML format.
        <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant
        """
        prompt = ""
        for msg in self.history:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    async def generate(self, user_input: str):
        # 1. Add user input to history
        self.history.append({"role": "user", "content": user_input})
        
        # 2. Format the full conversation for the model
        full_prompt = self._format_chat()
        
        request_id = f"{self.user_id}-{uuid.uuid4()}"
        sampling_params = SamplingParams(
            temperature=self.temp_setting, 
            max_tokens=512,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # 3. Stream from the engine
        results_generator = self.engine.generate(full_prompt, sampling_params, request_id)
        
        final_text = ""
        async for request_output in results_generator:
            final_text = request_output.outputs[0].text
            
        # 4. Save the model's response to history for the next turn
        self.history.append({"role": "assistant", "content": final_text})

        return final_text
    
    def get_vram_status(self):
        get_vram_status()

# --- Example Usage ---
async def main():
    alice = UserSession("Alice")
    
    # First turn
    resp1 = await alice.generate("My name is Alice. Remember that.")
    print(f"Bot: {resp1}")
    
    # Second turn (The model will remember the name)
    resp2 = await alice.generate("What is my name?")
    print(f"Bot: {resp2}")

if __name__ == "__main__":
    asyncio.run(main())
