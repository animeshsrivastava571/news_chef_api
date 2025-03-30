from langchain_core.tools import tool

@tool
def get_market_value(player_name: str) -> str:
    """Gets the current market value of a player."""
    fake_db = {
        "Lionel Messi": "€50 million",
        "Cristiano Ronaldo": "€30 million",
    }
    return fake_db.get(player_name, "Market value information not available.")

