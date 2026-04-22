import os, sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import uvicorn
import asyncio
from inference.image_txt_llm import UserSession  # Multimodal LLM model
from typing import Annotated    
from fastapi import FastAPI, Header, Request
from contextlib import asynccontextmanager
from fastmcp.utilities.lifespan import combine_lifespans
from functools import wraps
from pydantic import BaseModel, Field
import signal
import pdb
from io import BytesIO

# clean_shutdown decorator facilitates clean stoppage of inference server
#  in case the server has an error on client create or inference.
def clean_shutdown(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        async def shutdown(sig_name):
            print(f"\nReceived {sig_name}. Cleaning up...")
            # Access the engine through your model client session
            client_id = kwargs.get("client_id")
            request = kwargs.get('request')
            await request.app.state.llm_sessions[client_id].shared_engine.__del__() 
            # await model.shared_engine.__del__() 
            stop_event.set()

        result = await func(*args, **kwargs) 
        
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, 
                lambda s=sig: asyncio.create_task(shutdown(s.name))
            )

        return result
    return wrapper

# state sessions to persist llm client between api endpoint calls.
@asynccontextmanager
async def session_lifespan(app: FastAPI):
    # Initialize a registry for your sessions
    app.state.llm_sessions = {}
    yield
    # Cleanup logic (e.g., closing client connections)
    # for session in app.state.llm_sessions.values():
    #     await session.close()
    app.state.llm_sessions.clear()

#### Core functions ####
# Create inference client session defined by 'client_id'
@clean_shutdown
async def new_session(client_id: str, request, data: dict):
    try:
        system_prompt = data.get(
            'system_prompt', 
            "You are a concise assistant. Provide direct, information-dense answers only."
        )

        request.app.state.llm_sessions[client_id] = await UserSession.create(
                client_id = client_id,
                system_prompt= system_prompt,
            )
    except:
        return {"status": "failed"}
    return {"status": "ok"}

# Run inference for client context defined by 'client_id'
@clean_shutdown
async def inference(client_id: str, request: Request, data: str):
    try:
        prompt = data.get('prompt_str', None)
        image_file = data.get('image_file', None) # binary
        max_tokens = data.get('max_tokens', None) 

        model = request.app.state.llm_sessions[client_id]
        result = await model.inference(
                image_path=image_file, 
                prompt_str_=prompt,
                max_tokens=int(max_tokens),
                verbose=True,
            )
    except:
        return {"status": "failed"}
    return {"status": "ok", "result": result}

############ API ENDPOINTS ##################
# Create FastAPI app with MCP lifespan (required for session management)
api = FastAPI(lifespan=session_lifespan)

@api.get("/api/status")
def status():
    return {"status": "ok"}

@api.post("/api/create_session/{client_id}")
async def create_session(client_id: str, request: Request):
    data = await request.json()
    response = await new_session(client_id, request, data)
    return response

@api.post("/api/inference_on_session/{client_id}")
async def inference_on_session(client_id: str, request: Request):
    
    data = await request.json()
    result = await inference(client_id, request, data)
    return result

if __name__ == "__main__":
    uvicorn.run("llm_client_sessions:api", host="0.0.0.0", port=8000, reload=True)

    