from fastapi import APIRouter

router = APIRouter(tags=["healthcheck"])


@router.get("/")
async def health_check():
    return {"status_code": 200, "detail": "ok", "result": "working"}
