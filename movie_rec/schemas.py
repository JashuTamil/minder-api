import pydantic
from typing import List

class MovieType(pydantic.BaseModel):
    id: int
    name: str
    director: str
    cast: List[str]
    year: int
    description: str
    url: str

class FeedbackResponse(pydantic.BaseModel):
    likes: List[MovieType]
    dislikes: List[MovieType]