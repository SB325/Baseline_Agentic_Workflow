import os, sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import uvicorn
import asyncio
from inference.image_txt_llm import get_vram_status, UserSession  # Multimodal LLM model
from typing import Annotated    
from fastapi import FastAPI, Header, Request, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from functools import wraps
from pydantic import BaseModel, Field
import signal
import pdb
from io import BytesIO
from dotenv import load_dotenv
import json

load_dotenv()
MAX_CLIENTS = int(os.getenv("MAX_VLLM_CLIENTS"))

class SuccessResponse(BaseModel):
    status: str = "success"
    message: str | None = None
    data: dict | None = None

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
    code: str | None = None

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
            request = kwargs.get('request', None)
            if request:
                await request.app.state.llm_sessions[client_id].shared_engine.__del__() 
                stop_event.set()

        result = await func(*args, **kwargs) 
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, 
                lambda s=sig: asyncio.create_task(shutdown(s.name))
            )

        return result
    return wrapper

def clean_shutdown_small(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        async def shutdown(sig_name):
            print(f"\nReceived {sig_name}. Cleaning up...")
            # Access the engine through your model client session
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

    # Create default session to warm up the vLLM engine before accepting requests
    print("**********\nStarting up engine prior to client connections\n**********")
    try:
        default_client_id = "default"
        system_prompt = "You are a concise assistant. Provide direct, information-dense answers only."
        
        # Create the default session using UserSession.create directly
        app.state.llm_sessions[default_client_id] = await UserSession.create(
            client_id=default_client_id,
            system_prompt=system_prompt,
        )
        
        print(f"Default session '{default_client_id}' created successfully")
        print("**********\nvLLM engine Started!\n**********")
        yield
    except Exception as e:
        print(f"Failed to create default session: {str(e)}\nExiting.")
        sys.exit(0)
    finally:
        # Ensure cleanup of ALL sessions
        for session in app.state.llm_sessions.values():
            if hasattr(session, 'shared_engine'):
                await session.shared_engine.__del__()
        app.state.llm_sessions.clear()
    
    # Cleanup logic (e.g., closing client connections)
    # for session in app.state.llm_sessions.values():
    #     await session.close()
    app.state.llm_sessions.clear()

#### Core functions ####
# Create inference client session defined by 'client_id'
@clean_shutdown
async def new_session(session_data, request):
    try:
        data = json.loads(session_data)
        client_id = data.get('client_id')
        if not client_id:
            raise
        client_exists = request.app.state.llm_sessions.get(client_id, None)
        if not client_exists:
            n_clients = len(request.app.state.llm_sessions)
            if n_clients <= MAX_CLIENTS:
                system_prompt = data.get(
                    'system_prompt', 
                    "You are a concise assistant. Provide direct, information-dense answers only."
                )

                request.app.state.llm_sessions[client_id] = await UserSession.create(
                        client_id = client_id,
                        system_prompt= system_prompt,
                    )
                return JSONResponse(
                    status_code=status.HTTP_201_CREATED,
                    content={"status": "success", "message": f"Session '{client_id}' created successfully"}
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Cannot create duplicate clients."
                )
        else:
            return {"status": "Bad Request", "reason": "Cannot create duplicate clients."}
    except Exception as e:
        msg = f"Failed to create session with client_id {client_id}: {str(e.__dict__)}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=msg
        )

@clean_shutdown
async def remove_session(session_data, request):
    try:
        data = json.loads(session_data)
        client_id = data.get('client_id')
        if not client_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No Session ID found in request!"
            )
        client_exists = request.app.state.llm_sessions.get(client_id, None)
        if client_exists:
            try:
                del request.app.state.llm_sessions[client_id]
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to remove session with client_id {client_id}: {str(e)}"
                )

        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Session ID -{client_id}- does not exist.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )

# Run inference for client context defined by 'client_id'
@clean_shutdown
async def inference(session_data, request: Request):
    try:
        data = json.loads(session_data)
        client_id = data.get('client_id')
        if not client_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No Session ID found in request!"
            )
        client_exists = request.app.state.llm_sessions.get(client_id, None)
        if not client_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{client_id}' does not exist."
            )

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
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "data": {"result": result}}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@clean_shutdown_small
async def get_gpu_status():
    return get_vram_status()

############ API ENDPOINTS ##################
# Create FastAPI app with lifespan (required for session management)
api = FastAPI(
    lifespan=session_lifespan
    # root_path=os.getenv("ROOT_VLLM_PROXY_PREFIX", "vllm")
)

@api.get("/api/status")
def get_status():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "active_sessions": len(api.state.llm_sessions),
            "max_clients": MAX_CLIENTS
        }
    )

class sessionModel(BaseModel):
    client_id: str
    system_prompt: str | None = None
    image_file: str | None = None

class delSessionModel(BaseModel):
    client_id: str

class inferenceSessionModel(BaseModel):
    client_id: str
    prompt_str: str
    image_file: str | None = None
    max_tokens: int

@api.post("/api/create_session")
async def create_session(session_data: sessionModel, request: Request = None):
    data = session_data.model_dump_json()
    return await new_session(data, request)


@api.post("/api/delete_session")
async def delete_session(session_data: delSessionModel, request: Request):
    data = session_data.model_dump_json()
    return await remove_session(data, request)

@api.post("/api/inference_on_session")
async def inference_on_session(session_data: inferenceSessionModel, request: Request):
    data = session_data.model_dump_json()
    return await inference(data, request) 

@api.get("/api/get_gpu_status")
async def gpu_status():
    vram_stats = await get_gpu_status()
    return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "data": {"result": vram_stats}}
        )

if __name__ == "__main__":
    uvicorn.run("llm_client_sessions:api", host="0.0.0.0", port=8000, reload=True)

    