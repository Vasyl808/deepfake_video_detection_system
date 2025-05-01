from pydantic import BaseModel
from typing import List


class FrameResult(BaseModel):
    frame_number: int
    image: str
    