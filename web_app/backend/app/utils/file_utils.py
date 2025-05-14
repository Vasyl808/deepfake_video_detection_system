import re


def is_youtube_url(url: str) -> bool:
    return (
        re.match(r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=[\w-]+', url)
        or re.match(r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/[\w-]+', url)
    )