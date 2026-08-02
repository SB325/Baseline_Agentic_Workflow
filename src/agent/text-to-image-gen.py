from ..inference import triton_inference
import pdb

# LLM 
text_to_image_model = "stabilityai/stable-diffusion-xl-refiner-1.0"

class text_to_image():
    client = triton_inference(text_to_image_model)

    def filter_text(prompt: str):
        result = self.text_client.run_inference(prompt, to_file=True)
        
        if not result:
            print(f'Text to image inference failed!')

        return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Text-to-Image model inference',
                    description='This script accepts a prompt requesting an image and saves the image to file.',
                    epilog='by: SFB')
    parser.add_argument('prompt')

    args = parser.parse_args()
    input_prompt = args.prompt
    
    tti = text_to_image()
    result = tti.filter_text(input_prompt)
    pdb.set_trace()