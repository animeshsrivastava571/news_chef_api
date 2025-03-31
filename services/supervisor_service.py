from services.market_value_service import market_value_researcher_agent
from services.current_club_service import current_club_researcher_agent
from services.text_writer_service import text_writer_agent
from services.supervisor_agent import news_chef
from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph


class InputArticleState(TypedDict):
    article: str


class OutputFinalArticleState(TypedDict):
    final_article: str
    off_or_ontopic: str


class SharedArticleState(InputArticleState, OutputFinalArticleState):
    mentions_market_value: str
    mentions_current_club: str
    meets_100_words: str


def update_article_state(state: SharedArticleState) -> SharedArticleState:
    """Runs the supervisor agent to check what updates are needed."""
    if "off_or_ontopic" not in state:
        response = news_chef(state["article"])
        state["off_or_ontopic"] = response.off_or_ontopic
        state["mentions_market_value"] = response.mentions_market_value
        state["mentions_current_club"] = response.mentions_current_club
        state["meets_100_words"] = response.meets_100_words

    return state


def market_value_researcher_node(state: SharedArticleState) -> SharedArticleState:
    """Adds market value if missing."""
    response = market_value_researcher_agent.invoke({"article": state["article"]})
    state["article"] += f" {response['agent_output']}"
    state["mentions_market_value"] = "yes"
    return state


def current_club_researcher_node(state: SharedArticleState) -> SharedArticleState:
    """Adds current club if missing."""
    response = current_club_researcher_agent.invoke({"article": state["article"]})
    state["article"] += f" {response['agent_output']}"
    state["mentions_current_club"] = "yes"
    return state


def word_count_rewriter_node(state: SharedArticleState) -> SharedArticleState:
    """Expands article to at least 100 words if needed."""
    response = text_writer_agent.invoke({"article": state["article"]})
    state["article"] += f" {response['agent_output']}"
    state["final_article"] = response["agent_output"]
    state["meets_100_words"] = "yes"
    return state


def news_chef_decider(
    state: SharedArticleState,
) -> Literal[
    "market_value_researcher", "current_club_researcher", "word_count_rewriter", END
]:
    """Decides which agent should run next."""
    if state["off_or_ontopic"] == "no":
        return END
    if state["mentions_market_value"] == "no":
        return "market_value_researcher"
    elif state["mentions_current_club"] == "no":
        return "current_club_researcher"
    elif (
        state["meets_100_words"] == "no"
        and state["mentions_market_value"] == "yes"
        and state["mentions_current_club"] == "yes"
    ):
        return "word_count_rewriter"
    else:
        return END


workflow = StateGraph(
    SharedArticleState, input=InputArticleState, output=OutputFinalArticleState
)

workflow.add_node("supervisor", update_article_state)
workflow.add_node("market_value_researcher", market_value_researcher_node)
workflow.add_node("current_club_researcher", current_club_researcher_node)
workflow.add_node("word_count_rewriter", word_count_rewriter_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    news_chef_decider,
    {
        "market_value_researcher": "market_value_researcher",
        "current_club_researcher": "current_club_researcher",
        "word_count_rewriter": "word_count_rewriter",
        END: END,
    },
)

workflow.add_edge("market_value_researcher", "supervisor")
workflow.add_edge("current_club_researcher", "supervisor")
workflow.add_edge("word_count_rewriter", "supervisor")

app = workflow.compile()


def process_supervised_article(article: str) -> dict:
    """Runs the supervised agent workflow."""
    state = {"article": article}
    result = app.invoke(state)
    return {
        "original_article": article,
        "final_article": result["final_article"],
        "off_or_ontopic": result["off_or_ontopic"],
    }
