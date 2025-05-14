from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging

from app.core.config import app_config
from app.routes import videos, health_checks
from app.utils.file_manager import file_manager, clear_results_dir

import os
import warnings
import asyncio

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

os.makedirs(app_config.TEMP_DIR, exist_ok=True)
os.makedirs(app_config.RESULTS_DIR , exist_ok=True)

app = FastAPI(
    title="Video Analysis API",
    description="API для аналізу відеофайлів",
    version="1.0.0",
)

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
app.mount("/analyzed_frames", StaticFiles(directory=app_config.RESULTS_DIR ), name="analyzed_frames")

scheduler = AsyncIOScheduler()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Глобальна помилка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Виникла внутрішня помилка сервера. Спробуйте пізніше."},
    )


@app.on_event("startup")
async def startup_event():
    logger.info("Запуск сервера...")

    try:
        redis = await file_manager.get_redis()
        await redis.ping()
        logger.info("З'єднання з Redis встановлено успішно")
    except Exception as e:
        logger.error(f"Помилка з'єднання з Redis: {e}")

    try:
        count = await file_manager.clean_old_files()
        logger.info(f"Видалено {count} старих файлів при запуску")
    except Exception as e:
        logger.error(f"Помилка при очищенні файлів: {e}")

    scheduler.add_job(file_manager.clean_old_files, trigger="interval", minutes=15)

    scheduler.add_job(clear_results_dir, trigger="cron", hour=0, minute=0)
    scheduler.start()
    logger.info("Планувальник очищення файлів запущено")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Зупинка сервера...")
    scheduler.shutdown(wait=False)
    logger.info("Планувальник зупинено")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=app_config.host, port=app_config.port, reload=True)