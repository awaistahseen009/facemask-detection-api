from pydantic import BaseModel

class APIOutput(BaseModel):
    facemask: str
    time_elapsed: str