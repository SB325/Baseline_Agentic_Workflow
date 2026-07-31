import torch
from fastapi import FastAPI, Request
from inference import triton_inference
import uvicorn
import pdb

load_dotenv(override=True)
in_docker = os.getenv("INDOCKER")

### API ###
app = FastAPI(root_path="/transcriber/api/v1")
speech_client = triton_inference("openai/whisper-small")

@app.post("/transcribe")
async def transcribe_audio(request: Request):
    # 1. Read the raw binary data from the request body
    body = await request.body()
    
    # 2. Convert bytes back to a NumPy array
    # IMPORTANT: Use the same dtype (e.g., float32) as your client-side recording
    result = speech_client.run_inference(body)

    # 3. (Optional) Reshape if your recording was multi-channel
    # audio_data = audio_data.reshape(-1, 1) 
    
    return {"text": result["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)