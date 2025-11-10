from fastapi import Body, APIRouter, UploadFile, File
from fastapi import status
from fastapi.responses import FileResponse

from movie_rec.data_processing import load_feedback

router = APIRouter(prefix="/api/v1/get")

@router.get("/get_file/")
async def send_file(file: UploadFile = File(...)):

    feedback = load_feedback()

    return FileResponse(feedback, media_type="application/json", filename="feedback.json")

