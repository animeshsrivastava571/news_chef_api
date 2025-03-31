from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from services.market_value_service import process_market_value_request
from services.current_club_service import process_current_club_request
from services.text_writer_service import process_text_expansion_request

app = FastAPI()

# Request model
class ArticleRequest(BaseModel):
    article: str

@app.post("/market_value")
def market_value_endpoint(request: ArticleRequest):
    """API endpoint for retrieving player market value."""
    return process_market_value_request(request.article)

@app.post("/current_club")
def current_club_endpoint(request: ArticleRequest):
    """API endpoint for retrieving the current club of a player."""
    return process_current_club_request(request.article)

@app.post("/expand_text")
def expand_text_endpoint(request: ArticleRequest):
    """API endpoint for expanding text to at least 100 words."""
    return process_text_expansion_request(request.article)

if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)

