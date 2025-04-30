from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import app_config
from app.routes import videos, health_checks
from app.utils.file_utils import clean_old_files

import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.makedirs(app_config.TEMP_DIR, exist_ok=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(health_checks.router)

app.mount("/videos", StaticFiles(directory=app_config.TEMP_DIR), name="videos")

scheduler = BackgroundScheduler()
scheduler.add_job(func=clean_old_files, trigger="interval", hours=1)
scheduler.start()


@app.on_event("startup")
async def startup_event():
    await run_in_threadpool(clean_old_files)  


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown(wait=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=app_config.host, port=app_config.port, reload=True)
    