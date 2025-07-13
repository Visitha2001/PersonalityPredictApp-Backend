from fastapi import APIRouter, HTTPException
from controllers.controller import PredictionController
from models.schemas import PersonalityInput, PersonalityPrediction

router = APIRouter()
controller = PredictionController()

@router.post("/predict", response_model=PersonalityPrediction)
async def predict_personality(input_data: PersonalityInput):
    try:
        return controller.predict(input_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    return {"status": "healthy"}