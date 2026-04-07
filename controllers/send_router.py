from fastapi import APIRouter
from fastapi.responses import JSONResponse
from movie_rec.data_processing import *

from movie_rec.data_processing import load_feedback
from movie_rec.schemas import FeedbackResponse


router = APIRouter(prefix="/api/v1/get")

@router.get("/get_feedback/{user}")
async def send_feedback(user):
    print(user)
    raw_feedback = load_feedback(user)
    feedback = FeedbackResponse(**raw_feedback)
    return JSONResponse(content=feedback.model_dump_json(), media_type="application/json")

@router.get("/get_movies/{user}")
async def send_movies(user):
    print(user)
    result = router_function(user)

    return JSONResponse(content=result, media_type="application/json")
