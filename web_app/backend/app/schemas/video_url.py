from pydantic import BaseModel
from typing import List


class VideoURL(BaseModel):
    url: str
