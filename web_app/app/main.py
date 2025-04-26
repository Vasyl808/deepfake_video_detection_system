from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
import uuid
import re
from typing import Optional
from video_analyzer import VideoAnalyzer
from schemas import AnalysisResponse

from pytubefix import YouTube
from pytubefix.cli import on_progress

import time
from apscheduler.schedulers.background import BackgroundScheduler
import threading

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB in bytes
TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure temp directory exists

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoURL(BaseModel):
    url: str

# --- Analyzing files set and lock ---
analyzing_files = set()
analyzing_lock = threading.Lock()

def clean_old_files(directory=TEMP_DIR, max_age_hours=1):
    """Remove temporary files older than max_age_hours, skip files being analyzed"""
    now = time.time()
    with analyzing_lock:
        analyzing = set(analyzing_files)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if filename in analyzing:
                # skip files currently being analyzed!
                continue
            file_mod_time = os.path.getmtime(filepath)
            if now - file_mod_time > max_age_hours * 3600:
                try:
                    os.remove(filepath)
                    print(f"[CLEANUP] Removed old file: {filepath}")
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    startTime: float = Form(...),
    duration: float = Form(...)
):
    file_size = 0
    temp_file_path = os.path.join(TEMP_DIR, f"uploaded_{uuid.uuid4()}_{file.filename}")
    filename = os.path.basename(temp_file_path)

    try:
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл занадто великий. Максимальний розмір: {MAX_FILE_SIZE // (1024 * 1024)}MB"
                    )
                buffer.write(chunk)

        await file.seek(0)

        if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".webm", ".mkv")):
            raise HTTPException(status_code=400, detail="Непідтримуваний формат відео")

        # Позначаємо файл як "аналізується"
        with analyzing_lock:
            analyzing_files.add(filename)
        try:
            analyzer = VideoAnalyzer()
            result = analyzer.analyze(temp_file_path, start_time=int(startTime), duration=int(duration))
        finally:
            # Завершили аналіз — прибираємо з set
            with analyzing_lock:
                analyzing_files.discard(filename)
        return result

    finally:
        # Видаляємо файл, якщо він вже не аналізується
        with analyzing_lock:
            if filename not in analyzing_files and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

def is_youtube_url(url: str) -> bool:
    return (
        re.match(r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=[\w-]+', url)
        or re.match(r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/[\w-]+', url)
    )

@app.post("/download-from-url")
async def download_from_url(
    video_data: VideoURL,
    background_tasks: BackgroundTasks,
):
    url = video_data.url.strip()
    if not is_youtube_url(url):
        raise HTTPException(
            status_code=400,
            detail="URL не розпізнано. Дозволено лише YouTube відео."
        )

    unique_id = uuid.uuid4()
    temp_dir = os.path.join(TEMP_DIR, str(unique_id))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = yt.streams.get_highest_resolution()
        filename = f"{yt.title}-{yt.video_id}.mp4"
        temp_path = os.path.join(temp_dir, filename)
        stream.download(output_path=temp_dir, filename=filename)

        if os.path.getsize(temp_path) > MAX_FILE_SIZE:
            os.remove(temp_path)
            raise HTTPException(
                status_code=413,
                detail="Відео занадто велике. Максимальний розмір: 500MB"
            )

        final_filename = f"youtube_{unique_id}_{os.path.basename(temp_path)}"
        permanent_path = os.path.join(TEMP_DIR, final_filename)
        shutil.move(temp_path, permanent_path)

        # Запуск очищення у фоні
        background_tasks.add_task(clean_old_files)

        return JSONResponse({
            "video_url": f"/videos/{final_filename}",
            "filename": final_filename
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Помилка завантаження: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

app.mount("/videos", StaticFiles(directory=TEMP_DIR), name="videos")

# --- APScheduler: запуск очищення щогодини ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=clean_old_files, trigger="interval", hours=1)
scheduler.start()

@app.on_event("startup")
async def startup_event():
    clean_old_files()

# --- Завершення scheduler при зупинці ---
@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown(wait=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)