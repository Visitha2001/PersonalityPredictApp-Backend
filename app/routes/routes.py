from fastapi import APIRouter
from controllers.controller import predict_personality
from models.schemas import PersonalityInput

router = APIRouter()

@router.post("/predict", tags=["personality"])
async def predict(input_data: PersonalityInput):
    result = predict_personality(input_data)
    return {"personality": result}