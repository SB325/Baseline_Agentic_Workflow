import os, sys
import pdb
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import asyncio
from typing import Dict, Any, Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import anyio
import json
from anyio import create_task_group
import subprocess
from dotenv import load_dotenv

load_dotenv()
brave_search_service_name = os.getenv("BRAVE_SEARCH_SERVICE_NAME")
brave_search_port = int(os.getenv("BRAVE_SEARCH_PORT"))

# Web Search Tool call argument model
class WebSearchArguments(BaseModel):
    query: str = Field(..., description="The main search query to look up on the web.")
    count: Optional[int] = Field(default=10, description="The number of results to return.")

# Schema for validation
TOOL_SCHEMAS: Dict[str, type[BaseModel]] = {
    "brave_web_search": WebSearchArguments,
}

def get_bravesearch_address():
    # Run a command and capture its stdout and stderr
    ip = subprocess.run(
        f"docker inspect --format='{{{{.NetworkSettings.Networks.homeserver.IPAddress}}}}' {brave_search_service_name}",
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode output as text (UTF-8 by default)
        shell=True           # Raise CalledProcessError if the command returns a non-zero exit code
    ).stdout.replace('\n', '')

    return f'http://{ip}:{brave_search_port}'

class BraveMCPClientInterface:
    def __init__(self, mode: str = "stdio", address_or_cmd: str = "npx", args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None):
        """
        Args:
            mode: "stdio" for local processes, "sse" for remote HTTP/Websocket URLs.
            address_or_cmd: HTTP URL for SSE mode, or binary command (like 'npx') for stdio mode.
            args: Runtime arguments for stdio mode (e.g., ['-y', '@modelcontextprotocol/server-brave-search']).
            env: Dictionary containing environment variables (e.g., {"BRAVE_API_KEY": "xxx"}).
        """
        self.mode = mode.lower()
        self.address_or_cmd = address_or_cmd
        self.args = args or []
        self.env = env or {}
        
        # State tracking
        self.session: Optional[ClientSession] = None
        self._exit_stack = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Exposes the real-time operational status of the MCP server connection."""
        return self._is_running and self.session is not None

    async def connect(self):
        """Establishes the connection lifecycle to the target MCP Server."""
        if self.is_running:
            return

        if self.mode == "sse":
            # Context manager tracking for Server-Sent Events (Web address)
            self._exit_stack = sse_client(url=self.address_or_cmd)
            read_stream, write_stream = await self._exit_stack.__aenter__()
        else:
            # Context manager tracking for standard local sub-processes
            server_params = StdioServerParameters(
                command=self.address_or_cmd,
                args=self.args,
                env=self.env
            )
            self._exit_stack = stdio_client(server_params)
            read_stream, write_stream = await self._exit_stack.__aenter__()

        # Instantiate session lifecycle
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        
        self._is_running = True
        print(f"[MCP] Client successfully connected via {self.mode.upper()}. Status: RUNNING.")

    async def disconnect(self):
        """Gracefully disconnects and shuts down tracking channels."""
        self._is_running = False
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None
        print("[MCP] Client disconnected. Status: STOPPED.")

    async def get_vllm_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Fetches tools from Brave MCP server and converts them 
        to OpenAI/vLLM compliant chat tool schemas.
        """
        if not self.is_running:
            raise RuntimeError("MCP Client is not connected. Call .connect() first.")

        mcp_manifest = await self.session.list_tools()
        vllm_tools = []

        for tool in mcp_manifest.tools:
            vllm_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        return vllm_tools

    async def execute_search(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Sends an execution call to the Brave MCP Server and pulls raw text strings 
        to injection directly back into the vLLM text generation context window.
        """
        if not self.is_running:
            return "Error: Search server is offline."

        print(f"[MCP] Routing query to tool '{tool_name}' with arguments: {arguments}")
        try:
            # Call tool natively across the established protocol stream
            result = await self.session.call_tool(name=tool_name, arguments=arguments)
            
            # Combine text components returned by the tool response
            text_responses = [content.text for content in result.content if hasattr(content, 'text')]
            return "\n".join(text_responses)
            
        except Exception as e:
            return f"Error executing Brave Search tool call: {str(e)}"

class AugmentationManager():
    def __init__(self):
        self.client = BraveMCPClientInterface(mode="sse", address_or_cmd=f"{get_bravesearch_address()}/sse")
        print(f"Server alive before initialization? {self.client.is_running}") # False
        self.vllm_ready_tools = None

    async def tool_connect(self):
        await self.client.connect()
        print(f"Server alive after initialization? {self.client.is_running}") # True

        self.vllm_ready_tools = await self.client.get_vllm_tools_schema()
        print(f"Successfully converted {len(self.vllm_ready_tools)} tools for llm.chat().")
        
    def show_tools(self):
        return self.vllm_ready_tools

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
        
        raw_web_context = await self.client.execute_search(target_tool, search_args)
        print("\n--- Gathered Search Payload for vLLM Context ---")
        print(raw_web_context[:300] + "...") # Preview output snippet

        return raw_web_context
            
    async def tool_disconnect(self):
        await self.client.disconnect()
        print(f"Server status post-teardown: {self.client.is_running}") # False

    async def __aenter__(self):
        """Triggers automatically when entering the 'async with' block."""
        await self.tool_connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Triggers automatically when exiting the 'async with' block, even on errors."""
        await self.tool_disconnect()

async def main():
    # aenter and aexit run tool_connect and disconnect automatically
    async with AugmentationManager() as manager:
        search_results = await manager.query(
            target_tool="brave_web_search", 
            search_args={"query": "vLLM performance optimization 2026"}
        )

if __name__ == "__main__":
    anyio.run(main)