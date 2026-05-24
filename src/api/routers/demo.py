from fastapi import APIRouter, Depends, Request

from src.api.quota import get_quota_status

router = APIRouter()


@router.get("/quota")
async def quota_status(request: Request, quota=Depends(get_quota_status)):
    return quota