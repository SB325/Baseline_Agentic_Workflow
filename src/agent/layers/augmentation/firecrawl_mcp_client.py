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
import copy
import codecs
from schemas.firecrawl_schemas import TOOL_SCHEMAS

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

class Firecrawl_MCP_Client():
    def __init__(self, tool_config, container_url: str = f"{get_firecrawl_address()}/mcp"):
        self.client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self.tool_config = tool_config
        self.toolbox = None
        self.url = container_url

    async def response(self, payload, headers):
        try:
            response = self.client.post(self.url, json=payload, headers=headers)

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
            print(f"Request failed: {e}")
            return []

    # Typically no LLM will ever recieve the product of this method, firecrawl's config not as important
    async def show_tools(self):
        """
        Sends a Streamable HTTP discovery request to inspect the 
        available tools, parameter limits, and definitions.
        """
        # 1. Modern Streamable HTTP routing requires explicit MCP headers
        headers = {
            "Accept": "application/json, text/event-stream", 
            "Content-Type": "application/json",
            # "Mcp-Method": "tools/list"  # Tells gateways/servers the exact primitive route
        }
        
        # 2. Construct standard JSON-RPC 2.0 Discovery Payload
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
            
        selected_tools = list(self.tool_config.keys())
        self.toolbox = await self.response(payload, headers)

        toolbox = self.toolbox['result']['tools']
        slim_result = copy.deepcopy(self.toolbox)
        slim_result['result']['tools'] = []

        for cnt, tool in enumerate(toolbox):
            if tool['name'] in selected_tools:
                slim_result['result']['tools'].append(tool)

        self.toolbox = slim_result

        return self.toolbox

    def parse_llm_response(self, llm_response: str):
        # takes llm response, checks it for a tool call, and if present, runs the tool and presents the MCP response
        response_data = json.loads(llm_response.read())

        choice = response_data["choices"][0]
        message = choice["message"]
        
        try:
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    func_data = tool_call["function"]
                    function_name = func_data["name"]
                    raw_arguments_str = func_data["arguments"]
                    
                    print(f"[VLLM ATTEMPT] Model generated tool call: '{function_name}'")
                    print(f"[VLLM PAYLOAD] Raw JSON String: {raw_arguments_str}")
                    
                    # Validate tool request object
                    schema_cls = TOOL_SCHEMAS.get(function_name)
                    if not schema_cls:
                        print(f"[VALIDATION FAILED] Unknown tool name encountered: {function_name}")
                        continue

                    try:
                        # Parse raw string JSON safely
                        parsed_json = json.loads(raw_arguments_str)
                        
                        # Enforce validation using Pydantic
                        validated_arguments = schema_cls(**parsed_json)
                        print("[VALIDATION PASSED] Tool call syntax and arguments conform fully to the JSON Schema!")
                        
                        self.query(target_tool=function_name, search_args=parsed_json)
                        print(f"[MCP RESPONSE] {result_string}")
                        
                    except json.JSONDecodeError:
                        print("[VALIDATION FAILED] Invalid JSON structure returned by vLLM.")
                    except ValidationError as e:
                        print(f"[VALIDATION FAILED] Parameter data type mismatches or missing required keys:\n{e}")
            else:
                print(f"[ASSISTANT DIRECT RESPONSE]: {message['content']}")
        
        except urllib.error.URLError as e:
            print(f"Connection to local vLLM instance failed: {e.reason}")

        return result_string

    async def query(self, target_tool: str, search_args: dict):
        # Typically you'd capture 'tool_call.function.name' and 'json.loads(tool_call.function.arguments)' from llm
        # target_tool = "brave_web_search"
        # search_args = {"query": "vLLM latest release news 2026"}
        
        headers_dict = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            # "method": "tools/call"
        }

        # 2. Re-verify payload structure
        payload_dict = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": target_tool,
                "arguments": {
                    "url": str(search_args['url']),
                    "formats": search_args['formats'],
                    'jsonOptions': search_args['jsonOptions'],
                }
            },
            "id": 2
        }

        web_scrape = await self.response(headers=headers_dict, payload=payload_dict)
        print("\n--- Gathered Search Payload for vLLM Context ---")

        return web_scrape
        
async def main(config_dict: dict):
    client = Firecrawl_MCP_Client(config_dict)
    search_results = []
    tools = list(config_dict.keys())
    for tool in tools:
        search_results.append(
                await client.query(
                target_tool=tool, 
                search_args={
                    "url": "https://docs.stripe.com/api/charges",
                    "formats": ['markdown'],
                    "onlyMainContent": True,      # 1. Drops sidebars, headers, footers, & nav links
                    "blockAds": True,             # 2. Stops heavy ad scripts and network tracking
                    "removeBase64Images": True,   # 3. Prevents heavy embedded image data strings
                    "excludeTags": ["img", "a"],  # 4. Manually drops standard image and link HTML tags
                    "timeout": 60000,   
                    "jsonOptions" : {
                        'prompt': 'Scrape the page. Get text only. Ignore links.'
                    },
                    }
            )
        )
    # pdb.set_trace()
    found_tools = await client.show_tools()
    print(f"Using {len(found_tools)} tools.")

    return search_results

# Execute discovery
if __name__ == "__main__":
    config_dict = {}
    config_file = "firecrawl_config.yaml"
    
    if Path(config_file).exists():
        yaml = YAML(typ='safe') # Targets YAML 1.2 strictly
        with open(config_file, "r") as f:
            config = yaml.load(f).get('functions', None)

    if not config:
        print(f"Config specifies no tool functions to include!\n\n" + \
            "Without a config yaml, the FULL suite of function " + \
            "calls will be provided to the LLM, *severely* limiting " + \
            "the context window!\n\n")

    results = anyio.run(main, config)

    print("Results:\n")

    if not results[0]:
        print("Nothing returned from query.")
    elif results[0]['result'].get('isError', None):
        print(f"Error in query. \n\n{results[0]['result']['isError']}")
    else:
        text = results[0]['result']['content'][0]['text']
        data = json.loads(text)
        # pdb.set_trace()
        if data.get('markdown', None):
            scraped_text = data['markdown']
            clean_markdown = codecs.decode(scraped_text, "unicode_escape")
            # pdb.set_trace()
            print(clean_markdown)
        elif data.get('json', None):
            scraped_text = data['json']['pageContent']
            flat_text = " ".join(scraped_text.split())
            # pdb.set_trace()
            print(json.dumps(flat_text, indent=2))
        else:
            print(f"Key value in results object not recognized. \n Keys: {results.keys()}")