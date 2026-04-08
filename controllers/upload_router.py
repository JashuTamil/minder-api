import json
from fastapi import Body, APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

from movie_rec.data_processing import save_feedback

router = APIRouter(prefix="/api/v1/upload")

@router.post("/send_feedback/{user}")
async def file_upload(user: str, data: dict = Body(...)):
    save_feedback(data, user)

    return JSONResponse(status_code=status.HTTP_200_OK, content='File uploaded successfully!')

