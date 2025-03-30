from pydantic import BaseModel

class ArticleRequest(BaseModel):
    article: str

class ArticleResponse(BaseModel):
    article: str
    agent_output: str
