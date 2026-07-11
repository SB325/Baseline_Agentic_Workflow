## Entity Extraction Agent ##################################
#      This agent extracts entities within a multi-sentence
#   block of text without a provided schema. 
#   The agent performs this through hierarchical categorization
#   from part of speech of a token down to it's more specific categorization.
#   For example,    [France] -> Noun -> Territory -> Country
#                   [Boeing 747] -> Noun -> Vehicle -> Aircraft

#   The state of each token in the sentence has the fields:
#   - Type, Index (1st character position in body), length (of token string)
    
#   The Type field contains subfields: 
#   - Part-of-Speech (PoS), Ordered List of Taxonomy

# The agent workflow is a cascade of looping subagents where the first 
# subagent determines "Is this token a Noun?" If no, go to next token.
# If yes, the next subagent is asked "Is this token a 1) Person, 2) Place, 
# 3) Thing, 4) Abstract Idea, or 5) OTHER?" The response to this prompt
# will be passed to three downstream subagents that will be asked the
# following 3 questions: 
# 1) "Is it a Common Noun or Proper Noun? Common nouns name general items (e.g., city, dog). 
#   Proper nouns name specific entities, are capitalized, and include 
#   names like London or Rover."
# 2) "Is it a Concrete Noun or Abstract Noun? Concrete nouns refer to physical things you 
#   can perceive with your senses (pizza, thunder). Abstract nouns 
#   represent intangible concepts (love, courage)."
# 3) "Is it a Countable Noun or Uncountable Noun? Countable nouns can be 
#   enumerated (e.g., one book, three books). Uncountable nouns represent 
#   masses or quantities that cannot be counted individually (water, sand).""

#  Entity tokens are specific, concrete representations of physical or conceptual 
#  people, places or things that can exist as no more than one single identity.
#  - Not entities: "city", "dog", "pizza", "thunder", "water", "sand". 
        # - can refer to one of several instances.
#  - Entities: "London", "Rover", "Hurricane Ida", "Lake Huron", "Mojave Desert"
        # - can only refer to one distinct instance. Proper Nouns
## Workflow ##
# Agent 1: Is it a Noun?
# Downstream Agent 2: Is it a common noun or proper noun?
# Branching downstream agent 3: Apply common noun label to token.
# Branching downstream agent 4: Apply proper noun label to token.
# Downstream agent 5 (from agent 4): Provide a category that this proper noun fits in. 

#################################################
from nodes.llm_inference import LLMInference
from typing import Annotated, Literal, Union
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import MessagesState # comes prebuilt with a list of AnyMessage objects and add_messages reducer
from langgraph.types import interrupt
from dotenv import load_dotenv
import pdb
import asyncio

load_dotenv()
# vllm_service_name = os.getenv("VLLM_SERVICE_NAME")

llm = LLMInference()

class InputState(MessagesState):
    session_id: str
    system_prompt: str
    user_input: str

class OutputState(MessagesState):
    agent_response: str

class OverallState(MessagesState):
    session_id: str
    system_prompt: str
    user_input: str
    agent_output: str

async def create_and_run_inference(session_id, system_prompt, persist_session: bool = False):
    await llm.create_session(
            client_id = session_id,
            system_prompt = system_prompt,
        )
    response = await llm.inference(
                client_id = session_id,
                prompt_str = system_prompt,
            )
    if not persist_session:
        await llm.delete_session(session_id)

    return response

def run_inference_on_session(state: InputState) -> OutputState:
    output = asyncio.run(
        create_and_run_inference(
            state['session_id'], 
            state['system_prompt'],
        )
    )
    return {'agent_response': output}

# If the LLM determines more agent loops are needed, it returns an updated InputInterractionState.
# If the LLM determines the task is complete, it mutates the payload into OutputInterractionState, 
# signaling to the graph that it can route to an end node.
# def llm_orchestrator(
#         input_state: InputInterractionState,
#         output_state: OutputInterractionState,
#     ) -> Union[InputInterractionState, OutputInterractionState]:
#     return {"agent_output": state["bar"] + " Lance"}


# def inbound_route_decision(
#         input_state: InputInterractionState
#     ) -> Literal[
#         "text_to_image_generator", 
#         "funny_joke_generator",
#         "inappropriate_input_handler",
#         "invalid_input_handler",
#         ]:
#     """Route based on whether input requests an image or a joke."""
#     if input_state["joke_request"]:
#         return Command(
#             # state update
#             update={"foo": "bar"},
#             # control flow
#             goto="funny_joke_generator"
#         )
#     elif input_state["image_request"]:
#         return Command(
#             # state update
#             update={"foo": "bar"},
#             # control flow
#             goto="text_to_image_generator"
#         )
#     elif input_state["nsfw_input"]: # inappropriate input detected
#         return "inappropriate_input_handler"
#     else input_state["invalid_input"]: # not joke or image request
#         return "invalid_input_handler"

def user_interface(
        conversation: OverallState
    ) -> OverallState:
    if not conversation["user_input"]:
        # Use interrupts for Human In the Loop support
        conversation["user_input"] = interrupt("Please give your name.\n")
        conversation["system_input"] = interrupt("Enter the system prompt.\n")
    else:
        # continue conversation with llm
        print( conversation['agent_output'] )

        conversation["user_input"] = interrupt("Your reply.\n")

    return {conversation["user_input"]}

def llm_server(
        conversation: OverallState
    ) -> OverallState:
    if len(conversation["user_input"]) == 1:
        # send system_prompt and client_id to create_llm_session_mcp mcp endpoint
        client_id = conversation["user_input"]
        system_input = conversation["system_input"]
        # llm client call to server here
    else:
        # continue conversation with llm
        prompt = conversation["user_input"]
        conversation["agent_output"] = # llm client call to server here
        
    return {conversation["agent_output"]}

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)

# Chatbot agent
builder.add_node(user_interface)
builder.add_node(llm_server)

# builder.add_node(speech_to_text)
# builder.add_node(nsfw_input_filter)
# builder.add_node(image_or_joke_filter)
# builder.add_node(llm_orchestrator)
# builder.add_node(inappropriate_input_handler)
# builder.add_node(invalid_input_handler)
# builder.add_node(text_to_image_generator)
# builder.add_node(funny_joke_generator)
# builder.add_node(nsfw_output_filter)

# Inbound edges
builder.add_edge(START, "user_interface")
builder.add_edge("user_interface", "llm_server")

# builder.add_edge(START, "speech_to_text")
# builder.add_edge("speech_to_text", "nsfw_input_filter")
# builder.add_edge("nsfw_input_filter", "image_or_joke_filter")
# builder.add_edge("image_or_joke_filter", "llm_orchestrator")
# builder.add_conditional_edges("llm_orchestrator", route_decision)

# Outbound edges
builder.add_edge("llm_server", "user_interface")
builder.add_edge("text_to_speech", END)

# builder.add_edge("text_to_image_generator", "nsfw_output_filter")
# builder.add_edge("funny_joke_generator", "nsfw_output_filter")
# builder.add_edge("nsfw_output_filter", "llm_orchestrator")
# builder.add_edge("nsfw_output_filter", "llm_orchestrator")
# builder.add_edge("llm_orchestrator","text_to_speech")
# builder.add_edge("text_to_speech", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})