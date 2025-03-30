from typing import TypedDict, List, Literal, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from .tools.current_club_tool import get_current_club

from dotenv import load_dotenv

load_dotenv()

class InputState(TypedDict):
    article: str

class OutputState(TypedDict):
    agent_output: str

class OverallState(InputState, OutputState):
    messages: Annotated[List[BaseMessage], add]

# Initialize model and bind tools
tools_current_club = [get_current_club]
model_current_club = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools_current_club)

def call_model_current_club(state: OverallState):
    """Calls the LLM model to determine the current club of a player."""
    local_messages = state.get("messages", [])

    if not local_messages:
        local_messages.append(HumanMessage(content=state["article"]))

    system_message = SystemMessage(
        content="""You are an Football assistant agent tasked with determining the current club of a player.
        Check for any spelling mistakes in the inputs eg: if the question says something about Ronaldo, infer that the 
        the player name is Cristiano Ronaldo similarly for Messi etc.
        If the club is mentioned, return it. Otherwise, return 'Current club information not available.'"""
    )
    response = model_current_club.invoke([system_message] + local_messages)
    print(f"state is {state}")
    state["agent_output"] = response.content
    state["messages"] = local_messages + [response]
   
    return state

def should_continue(state: OverallState) -> Literal["tools", END]:
    """Decides whether to execute tools or end execution."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# Define agent workflow
def create_current_club_agent():
    current_club_graph = StateGraph(OverallState)
    current_club_graph.add_node("call_model_current_club", call_model_current_club)
    current_club_graph.add_node("tools", ToolNode(tools_current_club))

    current_club_graph.add_edge(START, "call_model_current_club")
    current_club_graph.add_conditional_edges("call_model_current_club", should_continue)
    current_club_graph.add_edge("tools", "call_model_current_club")

    return current_club_graph.compile()

# Compile the agent
current_club_researcher_agent = create_current_club_agent()

# Generate and save workflow visualization
image_data = current_club_researcher_agent.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
with open('/Users/animeshsrivastava/Documents/GitHub/AI_Agents_learning_Animesh/Supervised_Agents/news_chef_api/services/images/current_club_researcher_agent.png', 'wb') as f:
    f.write(image_data)

def process_current_club_request(article: str) -> dict:
    """Runs the current club agent workflow."""
    state = {"article": article, "messages": []}
    result = current_club_researcher_agent.invoke(state)
    return {"article": article, "agent_output": result["agent_output"]}
