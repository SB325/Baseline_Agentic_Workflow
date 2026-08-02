from ..inference import triton_inference
import pdb

# LLM 
nsfw_text_model = "TostAI/nsfw-text-detection-large"
nsfw_image_model = "Falconsai/nsfw_image_detection"

class nsfw_filter():
    text_client = triton_inference(nsfw_text_model)
    iamge_client = triton_inference(nsfw_image_model)

    def filter_text(prompt: str):
        result = self.text_client.run_inference(prompt)
        
        return result

    def filter_image(prompt: str):
        result = self.image_client.run_inference(prompt)
        
        return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='NSFW (Not Safe For Work) Filter model',
                    description='This script accepts a text prompt or image and responds with a 0 or 1 \
                        suggesting whether or not an image is safe for work (innapropriate or not).',
                    epilog='by: SFB')
    parser.add_argument('prompt',
                    help='either a prompt string or a relative/absolute file path to an image.',
                    ) 

    args = parser.parse_args()
    input_prompt = args.prompt
    
    filter_ = nsfw_filter()
    if os.path.isfile(input_prompt):
        result = filter_.filter_image(input_prompt, to_file=True)
    else:
        result = filter_.filter_text(input_prompt)

    status = "Not Safe For Work" if result>0 else "Safe for Work"  
    
    print(f"This prompt or file is {status}")
    pdb.set_trace()