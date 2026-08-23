import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import anyio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from mcp import ClientSession
from mcp.client.sse import sse_client
import subprocess
from dotenv import load_dotenv
from enum import Enum   
from ruamel.yaml import YAML
import httpx
import json

load_dotenv()
firecrawl_service_name = os.getenv("FIRECRAWL_SERVICE_NAME")
firecrawl_port = int(os.getenv("FIRECRAWL_SEARCH_PORT"))

def get_firecrawl_address():
    # Run a command and capture its stdout and stderr
    ip = subprocess.run(
        f"docker inspect --format='{{{{.NetworkSettings.Networks.homeserver.IPAddress}}}}' {firecrawl_service_name}",
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode output as text (UTF-8 by default)
        shell=True           # Raise CalledProcessError if the command returns a non-zero exit code
    ).stdout.replace('\n', '')

    return f'http://{ip}:{firecrawl_port}'

def fetch_container_tool_definitions(container_url: str = f"{get_firecrawl_address()}/mcp"):
    """
    Sends a Streamable HTTP discovery request to inspect the 
    available tools, parameter limits, and definitions.
    """
    # 1. Modern Streamable HTTP routing requires explicit MCP headers
    # pdb.set_trace()
    headers = {
        "Accept": "application/json, text/event-stream", 
        "Content-Type": "application/json",
        "Mcp-Method": "tools/list"  # Tells gateways/servers the exact primitive route
    }
    
    # 2. Construct standard JSON-RPC 2.0 Discovery Payload
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    try:
        with httpx.Client() as client:
            with client.stream("POST", container_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "")
                
                # CASE 1: Standard immediate JSON response (e.g., tools/list)
                if "application/json" in content_type:
                    response.read() # Consume response text
                    return response.json()
                
                # CASE 2: The server returned an SSE Text Stream (e.g., firecrawl_crawl)
                elif "text/event-stream" in content_type:
                    full_stream_text = ""
                    aggregated_json_rpc = {}
                    
                    # Read the response line-by-line
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        # SSE streams prefix data payloads with "data: "
                        if line.startswith("data:"):
                            clean_json_str = line.replace("data:", "").strip()
                            
                            try:
                                chunk_data = json.loads(clean_json_str)
                                # Streamable HTTP wraps internal chunks inside JSON-RPC structures
                                if "result" in chunk_data:
                                    return chunk_data
                                aggregated_json_rpc = chunk_data
                            except json.JSONDecodeError:
                                # Fallback if your specific container build outputs raw strings instead of JSON patches
                                full_stream_text += clean_json_str
                    
                    # Return the constructed dictionary or fallback structure
                    return aggregated_json_rpc if aggregated_json_rpc else {"error": "Raw text fallback", "data": full_stream_text}
                
                else:
                    raise ValueError(f"Unsupported content type returned by server: {content_type}")
            
    except Exception as e:
        print(f"Handshake failed: {e}")
        return []

# Execute discovery
if __name__ == "__main__":
    result = fetch_container_tool_definitions()
    toolbox = result['result']['tools']
    pdb.set_trace()
    # for tool in toolbox:
    #     name = tool['name']
    #     description = tool['description']
    #     properties = tool['inputSchema']['properties']
    #     for property in properties:
    #         type = property['type']
    #         # if enum is a key, need validataion
    #     required_field = tool['inputSchema']['required']
    #     additionalProperties = tool['inputSchema']['additionalProperties']
        
    # result['result']['tools'] = list of tools
    # print(json.dumps(result, indent=2))
