from fastapi import FastAPI
from routes.routes import router as prediction_router

app = FastAPI(
    title="Personality Prediction API",
    description="API to predict if someone is Introvert or Extrovert",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*" , "http://localhost:5173/*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Personality Prediction Service",
        "docs": "/docs"
    }