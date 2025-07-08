from fastapi import FastAPI
from controllers.controller import initialize_models
from routes.routes import router
import uvicorn

app = FastAPI(
    title="Personality Prediction API",
    description="Predicts if a person is Introvert or Extrovert",
    version="1.0.0"
)

# Include routes
app.include_router(router, prefix="/api/v1")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    try:
        initialize_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)