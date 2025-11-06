from fastapi import Body, APIRouter, UploadFile, File
from fastapi import status
from fastapi.responses import FileResponse
from main import app

router = APIRouter(prefix="/api/v1/get")

@app.get("/get_file/")
async def send_file(file: UploadFile = File(...)):
    file_path = "/minder-api/movie_rec/user_data/feedback.json"

    return FileResponse(file_path, media_type="application/json", filename="feedback.json")

