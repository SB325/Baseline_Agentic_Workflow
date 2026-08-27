from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client 
from typing import Dict, Any, Optional, List
import subprocess
from schemas.bravesearch_schemas import TOOL_SCHEMAS
import os, sys
from dotenv import load_dotenv
import pdb

load_dotenv()
brave_search_service_name = os.getenv("BRAVE_SEARCH_SERVICE_NAME")
brave_search_port = int(os.getenv("BRAVE_SEARCH_PORT"))

def get_bravesearch_address():
    # Run a command and capture its stdout and stderr
    ip = subprocess.run(
        f"docker inspect --format='{{{{.NetworkSettings.Networks.homeserver.IPAddress}}}}' {brave_search_service_name}",
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode output as text (UTF-8 by default)
        shell=True           # Raise CalledProcessError if the command returns a non-zero exit code
    ).stdout.replace('\n', '')

    return f'http://{ip}:{brave_search_port}'

class BraveMCPClient:
    def __init__(self, mode: str = "sse", 
        address_or_cmd: str = f"{get_bravesearch_address()}/mcp", 
        args: Optional[List[str]] = None, 
        env: Optional[Dict[str, str]] = None,
        tool_config: Optional[List[dict]] = {},
        ):
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
        self.tool_config = tool_config['brave_config']
        # State tracking
        self.session: Optional[ClientSession] = None
        self._exit_stack = None
        self.is_running = False

    # @property
    # def is_running(self) -> bool:
    #     """Exposes the real-time operational status of the MCP server connection."""
    #     return self.is_running and self.session is not None

    async def connect(self):
        """Establishes the connection lifecycle to the target MCP Server."""
        if self.is_running:
            return

        if self.mode == "sse":
            # Context manager tracking for Server-Sent Events (Web address)
            self._exit_stack = streamablehttp_client(self.address_or_cmd)
            read_stream, write_stream, _ = await self._exit_stack.__aenter__()
        else:
            # Context manager tracking for standard local sub-processes
            server_params = StdioServerParameters(
                command=self.address_or_cmd,
                args=self.args,
                env=self.env
            )
            self._exit_stack = streamablehttp_client(server_params)
            read_stream, write_stream, _ = await self._exit_stack.__aenter__()

        # Instantiate session lifecycle
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        
        self.is_running = True
        print(f"[MCP] Client successfully connected via {self.mode.upper()}. Status: RUNNING.")

    async def disconnect(self):
        """Gracefully disconnects and shuts down tracking channels."""
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None
        self.is_running = False
        print("Bravesearch [MCP] Client disconnected. Status: STOPPED.")

    async def get_vllm_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Fetches tools from Brave MCP server and converts them 
        to OpenAI/vLLM compliant chat tool schemas.
        """
        if not self.is_running:
            raise RuntimeError("Bravesearch MCP Client is not connected. Call .connect() first.")

        # Get all tools from MCP server
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

        # downselect tool list and properties to limit context window consumption
        if self.tool_config:
            config_tools = list(self.tool_config.keys())
            # Downselect tools
            toolbox = [v for v in vllm_tools if v['function']['name'] in config_tools]
            
            # Downselect search properties
            for tool in toolbox:
                name = tool['function']['name']
                base_properties = list(tool['function']['parameters']['properties'].keys())
                drop_properties = []
                for field in base_properties:
                    # If default field is not in config field set, drop the field
                    if field not in self.brave_config.get(name).keys():
                        drop_properties.append(field)
                [tool['function']['parameters']['properties'].pop(field) for field in drop_properties]
            
            self.toolbox = toolbox
            
        return toolbox

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