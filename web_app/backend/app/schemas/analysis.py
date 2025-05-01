from pydantic import BaseModel
from typing import List
from app.schemas.frames import FrameResult


class SequenceAnalysis(BaseModel):
    classification: List[float]
    is_fake: bool
    explanation: str
    frames: List[FrameResult]
    gradcam: List[FrameResult]


class AnalysisResponse(BaseModel):
    sequences: List[SequenceAnalysis]
