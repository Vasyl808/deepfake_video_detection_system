from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from starlette import status
import os
from video_analyzer import VideoAnalyzer
from schemas import AnalysisResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://3000-vasyl808-fastapiapp-yey39s72moa.ws-eu118.gitpod.io",
    "http://localhost:3000",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Непідтримуваний формат відео")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        analyzer = VideoAnalyzer()
        result = analyzer.analyze(temp_file_path)
    finally:
        os.remove(temp_file_path)
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)