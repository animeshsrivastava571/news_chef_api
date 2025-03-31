from typing import TypedDict, List, Literal, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from .tools.market_value_tool import get_market_value

from dotenv import load_dotenv

load_dotenv()

class InputState(TypedDict):
    article: str


class OutputState(TypedDict):
    agent_output: str


class OverallState(InputState, OutputState):
    messages: Annotated[List[BaseMessage], add]

# Define overall state
class OverallState(InputState, OutputState):
    messages: Annotated[List[BaseMessage], add]

# Initialize model and bind tools
tools2 = [get_market_value]
model2 = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools2)

def call_model_market_value(state: OverallState):
    """Calls the LLM model to determine market value."""
    local_messages = state.get("messages", [])

    #print(f"Inside the call model market value function with input message {local_messages}")
    
    if not local_messages:
        local_messages.append(HumanMessage(content=state["article"]))

    system_message = SystemMessage(
        content="""You are an Football Assistant agent tasked with determining the market value of a player.
        You have 2 jobs:
        - Infer the player name from the input query. Note that the player name might not be exact eg. if its Ronaldo, it could be Cristiano Ronaldo
        if its Messi then its Lionel Messi and so on.
        - Use the player name to fetch the market value from the tool. 
        - If the market value is available, return it. Otherwise, return 'Market value information not available.
        - **Do not create your own responses. Use the tool to get the market value. 'Market value information not available'.**
       
        """
    )
    response = model2.invoke([system_message] + local_messages)

    state["agent_output"] = response.content
    state["messages"] = local_messages + [response]

    print(f"state is {state}")
    return state

def should_continue(state: OverallState) -> Literal["tools", END]:
    """Decides whether to execute tools or end execution."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# Define agent workflow
def create_market_value_agent():
    market_value_graph = StateGraph(OverallState)
    market_value_graph.add_node("call_model_market_value", call_model_market_value)
    market_value_graph.add_node("tools", ToolNode(tools2))
    
    market_value_graph.add_edge(START, "call_model_market_value")
    market_value_graph.add_conditional_edges("call_model_market_value", should_continue)
    market_value_graph.add_edge("tools", "call_model_market_value")
    
    return market_value_graph.compile()

# Compile the agent
market_value_researcher_agent = create_market_value_agent()

#Create a figure
image_data = market_value_researcher_agent.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
# Save the image to a file
with open('/Users/animeshsrivastava/Documents/GitHub/AI_Agents_learning_Animesh/Supervised_Agents/news_chef_api/services/images/market_value_researcher_agent.png', 'wb') as f:
    f.write(image_data)

def process_market_value_request(article: str) -> dict:
    """Runs the market value agent workflow."""
    state = {"article": article, "messages": []}
    result = market_value_researcher_agent.invoke(state)
    return {"article": article, "agent_output": result["agent_output"]}
