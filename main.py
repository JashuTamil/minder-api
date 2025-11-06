import fastapi
import uvicorn
from controllers import send_router, upload_router

app = fastapi.FastAPI()
app.include_router(upload_router.router)
app.include_router(send_router.router)

@app.get("/")
async def root():
    return{"message": "Was poppin?"}

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', workers=1)