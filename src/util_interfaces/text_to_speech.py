from ..inference import triton_inference

# LLM 
model = "OpenMOSS-Team/MOSS-TTSD-v1.0"

client = triton_inference(model)

def text_to_speech(prompt: str):
    full_prompt = prompt
    result = client.run_inference(full_prompt, to_file=True)
    response = {'returned_value': result}
        
    return response

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Text to Speech Model',
                    description='This script accepts a prompt and returns a speech file transcription.',
                    epilog='by: SFB')
    parser.add_argument('prompt')

    args = parser.parse_args()
    input_prompt = args.prompt
    
    speech = text_to_speech(input_prompt)
    pdb.set_trace()
    
    