from pydantic import BaseModel
from typing import Optional

class PersonalityInput(BaseModel):
    Time_spent_Alone: float
    Social_event_attendance: int
    Going_outside: int
    Friends_circle_size: int
    Post_frequency: int
    Stage_fear: str
    Drained_after_socializing: str

class PersonalityPrediction(BaseModel):
    prediction: str
    confidence: float
    indicators: dict