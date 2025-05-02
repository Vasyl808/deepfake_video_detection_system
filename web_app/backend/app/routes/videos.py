from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.services.video_analyzer import video_analyzer
from app.schemas.analysis import AnalysisResponse
from app.schemas.video_url import VideoURL
from app.core.config import app_config
from app.utils.file_manager import file_manager
from app.utils.file_utils import is_youtube_url

import os
import uuid
import shutil
from pytubefix import YouTube
from pytubefix.cli import on_progress

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    startTime: float = Form(...),
    duration: float = Form(...)
):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".webm", ".mkv")):
        raise HTTPException(status_code=400, detail="Непідтримуваний формат відео")
    
    try:
        temp_file_path, filename = await file_manager.save_upload_file(file)
        try:
            result = video_analyzer.analyze(temp_file_path, start_time=int(startTime), duration=int(duration))
            return result
        finally:
            await file_manager.release_file(filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка аналізу відео: {str(e)}")


@router.post("/download-from-url")
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
    temp_dir = os.path.join(app_config.TEMP_DIR, str(unique_id))
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = yt.streams.get_highest_resolution()
        filename = f"{yt.title}-{yt.video_id}.mp4"
        temp_path = os.path.join(temp_dir, filename)
        
        stream.download(output_path=temp_dir, filename=filename)
        
        if os.path.getsize(temp_path) > app_config.MAX_FILE_SIZE:
            os.remove(temp_path)
            raise HTTPException(
                status_code=413,
                detail="Відео занадто велике. Максимальний розмір: 500MB"
            )
        
        final_filename = f"youtube_{unique_id}_{os.path.basename(temp_path)}"
        permanent_path = os.path.join(app_config.TEMP_DIR, final_filename)
        
        shutil.move(temp_path, permanent_path)
        
        await file_manager.register_file(final_filename, status="done")
        
        background_tasks.add_task(file_manager.clean_old_files)
        
        return JSONResponse({
            "video_url": f"/videos/{final_filename}",
            "filename": final_filename
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Помилка завантаження: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
