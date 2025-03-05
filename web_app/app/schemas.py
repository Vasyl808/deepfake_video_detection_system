from pydantic import BaseModel
from typing import List


class FrameResult(BaseModel):
    frame_number: int
    image: str


class SequenceAnalysis(BaseModel):
    classification: List[float]
    is_fake: bool
    explanation: str
    frames: List[FrameResult]
    gradcam: List[FrameResult]


class AnalysisResponse(BaseModel):
    sequences: List[SequenceAnalysis]