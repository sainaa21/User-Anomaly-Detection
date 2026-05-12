from pydantic import BaseModel

class SessionData(BaseModel):

    user_id: int

    typing_speed: float
    avg_key_delay: float
    click_rate: float
    mouse_speed: float
    session_duration: float
    idle_time: float