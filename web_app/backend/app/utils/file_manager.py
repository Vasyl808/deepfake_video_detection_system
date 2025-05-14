import os
import time
import uuid
import redis
import logging
import aiofiles
import aioredis

from typing import Optional, List, Tuple
from fastapi import UploadFile, HTTPException

from app.core.config import app_config

logger = logging.getLogger(__name__)


class FileManager:
    def __init__(self, redis_url: str, temp_dir: str, max_file_size: int, ttl: int = 3600):
        self.redis_url = redis_url
        self.temp_dir = temp_dir
        self.max_file_size = max_file_size
        self.ttl = ttl
        self._ensure_temp_dir()
        self._redis = None
        
    def _ensure_temp_dir(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        
    async def get_redis(self) -> redis.Redis:
        if self._redis is None:
            try:
                self._redis = await aioredis.from_url(
                    self.redis_url, 
                    encoding="utf-8", 
                    decode_responses=True,
                    socket_timeout=5.0, 
                    socket_connect_timeout=5.0, 
                    retry_on_timeout=True 
                )
                await self._redis.ping()
                logger.info("Успішне з'єднання з Redis")
            except Exception as e:
                logger.error(f"Помилка з'єднання з Redis: {e}")
                raise HTTPException(status_code=500, detail="Помилка з'єднання з базою даних")
        return self._redis
        
    async def register_file(self, filename: str, status: str = "processing") -> None:
        try:
            redis_client = await self.get_redis()
            file_key = f"file:{filename}"
            await redis_client.hset(file_key, mapping={
                "status": status,
                "created_at": time.time()
            })
            await redis_client.expire(file_key, self.ttl)
        except Exception as e:
            logger.error(f"Помилка при реєстрації файлу {filename} в Redis: {e}")
        
    async def mark_file_completed(self, filename: str) -> None:
        try:
            redis_client = await self.get_redis()
            file_key = f"file:{filename}"
            await redis_client.hset(file_key, "status", "done")
        except Exception as e:
            logger.error(f"Помилка при оновленні статусу файлу {filename} в Redis: {e}")
        
    async def is_file_processing(self, filename: str) -> bool:
        try:
            redis_client = await self.get_redis()
            file_key = f"file:{filename}"
            status = await redis_client.hget(file_key, "status")
            return status == "processing"
        except Exception as e:
            logger.error(f"Помилка при перевірці статусу файлу {filename} в Redis: {e}")
            return False
        
    async def get_processing_files(self) -> List[str]:
        processing_files = []
        try:
            redis_client = await self.get_redis()
            
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(cursor, match="file:*", count=100)
                for key in keys:
                    status = await redis_client.hget(key, "status")
                    if status == "processing":
                        filename = key.split(":", 1)[1]
                        processing_files.append(filename)
                        
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Помилка при отриманні списку файлів з Redis: {e}")

        return processing_files
    
    def generate_temp_filename(self, original_filename: str, prefix: str = "uploaded") -> Tuple[str, str]:
        safe_original_filename = os.path.basename(original_filename)
        unique_id = uuid.uuid4()
        filename = f"{prefix}_{unique_id}_{safe_original_filename}"
        filepath = os.path.join(self.temp_dir, filename)
        return filepath, filename
        
    async def save_upload_file(self, file: UploadFile) -> Tuple[str, str]:
        temp_file_path, filename = self.generate_temp_filename(file.filename)
        
        file_size = 0
        try:
            async with aiofiles.open(temp_file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    file_size += len(chunk)
                    if file_size > self.max_file_size:

                        await buffer.close()
                        os.remove(temp_file_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Файл занадто великий. Максимальний розмір: {self.max_file_size // (1024 * 1024)}MB"
                        )
                    await buffer.write(chunk)
            
            await file.seek(0)
            await self.register_file(filename)
            return temp_file_path, filename
            
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if not isinstance(e, HTTPException):
                logger.exception(f"Помилка при збереженні файлу {filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Помилка при збереженні файлу: {str(e)}")
            raise
    
    async def clean_old_files(self) -> int:
        now = time.time()
        count = 0
        
        try:
            processing_files = await self.get_processing_files()
            
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if not os.path.isfile(filepath):
                    continue

                if filename in processing_files:
                    continue
                    
                file_mod_time = os.path.getmtime(filepath)
                if now - file_mod_time > self.ttl:
                    try:
                        os.remove(filepath)
                        count += 1
                        logger.info(f"[CLEANUP] Видалено старий файл: {filepath}")
                    except Exception as e:
                        logger.error(f"Помилка видалення {filepath}: {e}")
        except Exception as e:
            logger.error(f"Помилка при очищенні старих файлів: {e}")
                    
        return count
        
    async def release_file(self, filename: str) -> None:
        await self.mark_file_completed(filename)


async def clear_results_dir():
    dir_path = app_config.RESULTS_DIR
    logger.info(f"Почато щоденне очищення директорії результатів: {dir_path}")
    try:
        count = 0
        for root, dirs, files in os.walk(dir_path):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                    count += 1
                except Exception as e:
                    logger.error(f"Не вдалося видалити файл {name}: {e}")
            for name in dirs:
                full_dir = os.path.join(root, name)
                try:
                    if not os.listdir(full_dir):
                        os.rmdir(full_dir)
                except Exception as e:
                    logger.error(f"Не вдалося видалити директорію {full_dir}: {e}")
        logger.info(f"Щоденне очищення директорії результатів завершено, видалено {count} файлів.")
    except Exception as e:
        logger.error(f"Помилка при щоденному очищенні директорії результатів: {e}")


file_manager = FileManager(
    redis_url=app_config.REDIS_URL,
    temp_dir=app_config.TEMP_DIR,
    max_file_size=app_config.MAX_FILE_SIZE,
    ttl=app_config.FILE_TTL if hasattr(app_config, 'FILE_TTL') else 3600
)