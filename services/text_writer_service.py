from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv

load_dotenv()

class InputState(TypedDict):
    article: str

class OutputState(TypedDict):
    agent_output: str

class OverallState(InputState, OutputState):
    pass

# Initialize model
model_text_writer = ChatOpenAI(model="gpt-4o-mini")

def expand_text_to_100_words(state: OverallState):
    """Expands the input text to at least 100 words while maintaining its meaning."""
    human_message = HumanMessage(content=state["article"])
    system_message = SystemMessage(
        content="Expand the following text to be at least 100 words. Maintain the original meaning while adding detail."
    )
    response = model_text_writer.invoke([system_message, human_message])
    state["agent_output"] = response.content
    return state

# Define agent workflow
def create_text_writer_agent():
    text_writer_graph = StateGraph(OverallState)
    text_writer_graph.add_node("expand_text_to_100_words", expand_text_to_100_words)
    
    text_writer_graph.add_edge(START, "expand_text_to_100_words")
    text_writer_graph.add_edge("expand_text_to_100_words", END)

    return text_writer_graph.compile()

# Compile the agent
text_writer_agent = create_text_writer_agent()

# Generate and save workflow visualization
image_data = text_writer_agent.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
with open('/Users/animeshsrivastava/Documents/GitHub/AI_Agents_learning_Animesh/Supervised_Agents/news_chef_api/services/images/text_writer_agent.png', 'wb') as f:
    f.write(image_data)

def process_text_expansion_request(article: str) -> dict:
    """Runs the text expansion agent workflow."""
    state = {"article": article}
    result = text_writer_agent.invoke(state)
    return {"article": article, "agent_output": result["agent_output"]}
