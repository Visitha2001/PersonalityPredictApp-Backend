from pydantic import BaseModel

class PersonalityInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: bool
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: bool
    Friends_circle_size: float
    Post_frequency: float