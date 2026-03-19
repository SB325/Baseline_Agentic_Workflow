import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import triton_inference
import pdb
import argparse
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv(override=True)
model_directory = os.getenv("TRITON_MODEL_STORAGE")

# LLM 
model_namespace = 'meta_llama'
model = "Meta_Llama_3-8B_Instruct"
model_path = f"{model_directory}/models/{model_namespace}/{model}/1"

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        )
tokenizer.pad_token = tokenizer.eos_token

client = triton_inference(model=model_namespace, tokenizer=tokenizer)

def inference(prompt: str):
    result = client.run_inference(prompt)
        
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Llama Large Language Model',
                    description='This script accepts a prompt requesting and returns a respnose.',
                    epilog='by: SFB')
    parser.add_argument('prompt')

    args = parser.parse_args()
    input_prompt = args.prompt

    response = inference(input_prompt)
    print(response)
    pdb.set_trace()