from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
from starlette import status
import os
from video_analyzer import VideoAnalyzer
from schemas import AnalysisResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_video(
    file: UploadFile = File(...),
    startTime: float = Form(...),
    duration: float = Form(...)
):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Непідтримуваний формат відео")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        analyzer = VideoAnalyzer()
        # Ви можете використати значення startTime і duration для обрізання відео на сервері,
        # наприклад, змінити функцію extract_frames для прийому цих параметрів.
        result = analyzer.analyze(temp_file_path, start_time=int(startTime), duration=int(duration))
    finally:
        os.remove(temp_file_path)
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)