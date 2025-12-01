import fastapi
import uvicorn
from controllers import send_router, upload_router
from fastapi.middleware.cors import CORSMiddleware


app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']   
)

app.include_router(upload_router.router)
app.include_router(send_router.router)

@app.get("/")
async def root():
    return{"message": "Wus poppin?"}

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', workers=1)