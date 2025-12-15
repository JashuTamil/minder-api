import json
from fastapi import Body, APIRouter, UploadFile, File
from fastapi import status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1/upload")

@router.post("/send_feedback/")
async def file_upload(file: UploadFile = File(...)):
    with open(f"movie_rec/user_data/feedback.json", "wb") as buffer:
        json_bytes = file.file.read()
        json_data = json.loads(json_bytes.decode('utf-8'))
        file.file.close()

        buffer.write(json_data)

    return JSONResponse(status_code=status.HTTP_200_OK, content='File uploaded successfully!')

