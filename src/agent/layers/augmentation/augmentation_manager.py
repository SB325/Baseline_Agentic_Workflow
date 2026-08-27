import pdb
from pathlib import Path
# parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent)
# if parent_dir not in sys.path:
    # sys.path.insert(0, parent_dir)
from bravesearch_mcp_client import BraveMCPClient
from schemas.bravesearch_schemas import TOOL_SCHEMAS
from firecrawl_mcp_Client import Firecrawl_MCP_Client
import anyio
import json
from ruamel.yaml import YAML

class AugmentationManager():
    def __init__(self, config):
        self.brave_client = BraveMCPClient(config)
        self.firecrawl_client = Firecrawl_MCP_Client(config_dict)
        print(f"Server alive before initialization? {self.brave_client.is_running}") # False
        self.brave_config = config['brave_config']
        self.firecrawl_config = config['firecrawl_config']
        self.toolbox = None

    async def tool_connect(self):
        await self.brave_client.connect()
        print(f"Server alive after initialization? {self.brave_client.is_running}") # True

        self.brave_toolbox = await self.brave_client.get_vllm_tools_schema()
        print(f"Successfully converted {len(self.toolbox)} tools for llm.chat().")
        
    def show_tools(self):
        return self.brave_toolbox

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
                        
                        brave_result = self.query(target_tool=function_name, search_args=parsed_json)
                        full_results = {
                            'brave_results': [],
                            'full_results': [],
                        }
                        for result in brave_result:
                            full_results['brave_results'].append(result)
                            full_results['firecrawl_results'].append(
                                self.firecrawl_client.query(
                                    target_tool = "firecrawl_scrape",
                                    search_args = {
                                        "url": result['url'],
                                        "headers": {
                                            "Custom-Header-For-Target-Site": "value"
                                        }
                                    }
                                )
                            )
                        
                    except json.JSONDecodeError:
                        print("[VALIDATION FAILED] Invalid JSON structure returned by vLLM.")
                    except ValidationError as e:
                        print(f"[VALIDATION FAILED] Parameter data type mismatches or missing required keys:\n{e}")
            else:
                print(f"[ASSISTANT DIRECT RESPONSE]: {message['content']}")
        
        except urllib.error.URLError as e:
            print(f"Connection to local vLLM instance failed: {e.reason}")

        return full_results

    async def query(self, target_tool: str, search_args: dict):
        # Typically you'd capture 'tool_call.function.name' and 'json.loads(tool_call.function.arguments)' from llm
        # target_tool = "brave_web_search"
        # search_args = {"query": "vLLM latest release news 2026"}
        
        raw_web_context = await self.brave_client.execute_search(target_tool, search_args)
        print("\n--- Gathered Search Payload for vLLM Context ---")
        # print(raw_web_context[:300] + "...") # Preview output snippet

        # Parse url fields and get clean markdown of content from firecrawl API

        return raw_web_context
            
    async def tool_disconnect(self):
        await self.brave_client.disconnect()
        print(f"Server status post-teardown: {self.brave_client.is_running}") # False

    async def __aenter__(self):
        """Triggers automatically when entering the 'async with' block."""
        await self.tool_connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Triggers automatically when exiting the 'async with' block, even on errors."""
        await self.tool_disconnect()

async def main(config: dict):
    # aenter and aexit run tool_connect and disconnect automatically
    # ****** TOOL FUNCTIONS ($0.005 per request) ********
    # brave_web_search - 
    # brave_local_search - Searches for local businesses, restaurants, and physical places
    # brave_news_search - Pulls current news articles and breaking updates
    # brave_image_search - Finds images across the web, returning metadata like image URLs, 
    #           dimensions, and properties
    # brave_video_search - Searches for videos and returns rich metadata including titles, 
    #           descriptions, durations, and thumbnail URLs
    # brave_place_search - : Used in tandem with local search to gather deeper, targeted 
    #           details on specific geographic points of interest
    # brave_llm_context / brave_summarizer - Fetches heavily optimized web page content or 
    #           AI-generated summaries specifically formatted to reduce token usage and 
    #           ground LLM responses
    #
    # Configure available subset of tools and parameters for the LLM to select from using a
    #   .json file in the home directory. The augmentation layer will look for this file
    #   and use it to downselect from the complete set. LLM prompt requests for tools will
    #   be validated against this pydantic subset in case it discovers unselected tools/
    #   parameters from an online search.
    # ******************************************

    async with AugmentationManager(config) as manager:
        search_results = []
        tools = list(config['brave_config'].keys())
        for tool in tools:
            search_results.append(
                    await manager.query(
                    target_tool=tool, 
                    search_args={"query": "vLLM performance optimization 2026",
                        "count": 2}
                )
            )

        print(f"Using {len(manager.show_tools())} tools.")
    print(search_results)

if __name__ == "__main__":
    brave_config_file = {}
    firecrawl_config_file = {}
    brave_config_file = "brave_config.yaml"
    firecrawl_config_file = "firecrawl_config.yaml"

    if Path(brave_config_file).exists():
        yaml = YAML(typ='safe') # Targets YAML 1.2 strictly
        with open(brave_config_file, "r") as f:
            brave_config = yaml.load(f).get('functions', None)

    if Path(firecrawl_config_file).exists():
        yaml = YAML(typ='safe') # Targets YAML 1.2 strictly
        with open(firecrawl_config_file, "r") as f:
            firecrawl_config = yaml.load(f).get('functions', None)

    if not (brave_config or firecrawl_config):
        print(f"Missing configs for tool functions to include!\n\n" + \
            "Without a config yaml, the FULL suite of function " + \
            "calls will be provided to the LLM, *severely* limiting " + \
            "the context window!\n\n")

    anyio.run(main, {'brave_config': brave_config, 
            'firecrawl_config': firecrawl_config
            })