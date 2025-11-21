from fastapi import Body, APIRouter, UploadFile, File
from fastapi import status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/upload")

@router.post("/send_feedback/")
async def file_upload(file: UploadFile = File(...)):
    with open(f"movie_rec/user_data/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    return JSONResponse(status_code=status.HTTP_200_OK, content='File uploaded successfully!')

