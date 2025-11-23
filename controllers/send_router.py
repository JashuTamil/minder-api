from fastapi import Body, APIRouter, UploadFile, File
from fastapi import status
from fastapi.responses import JSONResponse

from movie_rec.data_processing import load_feedback
from movie_rec.schemas import FeedbackResponse

router = APIRouter(prefix="/api/v1/get")

@router.get("/get_feedback/")
async def send_feedback():
    raw_feedback = load_feedback()
    feedback = FeedbackResponse(**raw_feedback)
    print(feedback)
    return JSONResponse(content=feedback.model_dump_json(), media_type="application/json")

@router.get("/get_movies/")
async def send_movies():
    ...
