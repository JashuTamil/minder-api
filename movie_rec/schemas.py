import pydantic
from typing import List

class MovieType(pydantic.BaseModel):
    id: int
    title: str
    overview: str
    vote_average: float
    runtime: int
    poster_path: str
    cast: List[str]
    director: str
    release_date: str
    

class FeedbackResponse(pydantic.BaseModel):
    likes: List[MovieType]
    dislikes: List[MovieType]