import os, sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import uvicorn
import asyncio
from typing import Annotated    
from fastapi import FastAPI, Header, Request
from fastmcp import FastMCP
from contextlib import asynccontextmanager
from fastmcp.utilities.lifespan import combine_lifespans
from functools import wraps
from pydantic import BaseModel, Field
import signal
import pdb
from io import BytesIO
import random

def make_selection(choice: str):
    # Create a list containing 5 animals
    animals = ["Lion", "Elephant", "Penguin", "Dolphin", "Tiger"]
    countries = ["1, 3, 5, 7, 11"]
    
    if 'animal' in choice:
        selection = random.choice(animals)
    elif 'number' in choice:
        selection = random.choice(numbers)
        
    return selection

############ MCP ENDPOINTS ##################
class Selection(BaseModel):
    selection_type: str = Field(
        description="""Either the string 'animal', or 'number'.""")

# Create your MCP server
mcp = FastMCP("API Tools")

@mcp.tool
def johnnys_favorites_mcp(
            data_in: Selection
            ) -> str:
    """Provides the name of one of Johnny's favorite animals or numbers.

    Arguments:
    - data_in: dictionary object containing the key 'type' which can hold the string 'animal' or 'number'.
    Returns:
    A string representing one of Johnny's favorite animals or an integer representing one of his favorite numbers.
    """

    data = data_in.model_dump()
    return make_selection(data['selection_type'])

# Create the MCP ASGI app with path="/" since we'll mount at /mcp
mcp_app = mcp.http_app(path="/")

############ API ENDPOINTS ##################
# Create FastAPI app with MCP lifespan (required for session management)
api = FastAPI(lifespan=mcp_app.lifespan)

@api.get("/api/status")
def status():
    return {"status": "ok"}

@api.post("/api/johnnys_favorites")
async def johnnys_favorites(request: Request):
    data = await request.json()
    return make_selection(data['selection_type'])

# Mount MCP at /mcp
api.mount("/mcp", mcp_app)

if __name__ == "__main__":
    uvicorn.run("johnnys_favorites:api", host="0.0.0.0", port=8000, reload=True)

    