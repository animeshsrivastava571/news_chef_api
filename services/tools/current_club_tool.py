from langchain_core.tools import tool

@tool
def get_current_club(player_name: str) -> str:
    """Gets the current club of a player."""
    fake_db = {
        "Lionel Messi": "Inter Miami",
        "Cristiano Ronaldo": "Al Nassr",
    }
    return fake_db.get(player_name, "Current club information not available.")
