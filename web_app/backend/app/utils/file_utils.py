import os
import re
import threading
import time

from app.core.config import app_config

analyzing_files = set()
analyzing_lock = threading.Lock()


def clean_old_files(directory=app_config.TEMP_DIR, max_age_hours=1):
    now = time.time()
    with analyzing_lock:
        analyzing = set(analyzing_files)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if filename in analyzing:
                continue
            file_mod_time = os.path.getmtime(filepath)
            if now - file_mod_time > max_age_hours * 3600:
                try:
                    os.remove(filepath)
                    print(f"[CLEANUP] Removed old file: {filepath}")
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")


def is_youtube_url(url: str) -> bool:
    return (
        re.match(r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=[\w-]+', url)
        or re.match(r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/[\w-]+', url)
    )