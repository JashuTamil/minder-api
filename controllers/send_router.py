from fastapi import Body, APIRouter, UploadFile, File
from fastapi import status
from fastapi.responses import JSONResponse

from movie_rec.data_processing import load_feedback

router = APIRouter(prefix="/api/v1/get")

@router.get("/get_feedback/")
async def send_feedback():
    feedback = load_feedback()
    print(feedback)
    return JSONResponse(content=feedback, media_type="application/json")

@router.get("/get_movies/")
async def send_movies():
    ...
