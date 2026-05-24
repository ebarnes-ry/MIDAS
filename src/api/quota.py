from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Literal

import redis.asyncio as redis
from fastapi import Header, HTTPException, Request

QuotaKind = Literal["upload", "pipeline", "reason", "explain", "feedback"]

DEFAULT_LIMITS: dict[QuotaKind, int] = {
    "upload": int(os.getenv("MIDAS_DAILY_UPLOAD_LIMIT", "5")),
    "pipeline": int(os.getenv("MIDAS_DAILY_PIPELINE_LIMIT", "3")),
    "reason": int(os.getenv("MIDAS_DAILY_REASON_LIMIT", "0")),
    "explain": int(os.getenv("MIDAS_DAILY_EXPLAIN_LIMIT", "10")),
    "feedback": int(os.getenv("MIDAS_DAILY_FEEDBACK_LIMIT", "10")),
}

REDIS_URL = os.getenv("REDIS_URL")

redis_client = (
    redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    if REDIS_URL
    else None
)


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = now.date() + timedelta(days=1)
    reset_at = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return max(1, int((reset_at - now).total_seconds()))


def _ip_prefix(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip()

    if not ip and request.client:
        ip = request.client.host

    ip = ip or "unknown"

    if ":" in ip:
        return ":".join(ip.split(":")[:4])

    return ".".join(ip.split(".")[:3])


def _subject(request: Request, client_id: str | None) -> str:
    raw = f"{client_id or 'missing'}:{_ip_prefix(request)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _key(kind: QuotaKind, subject: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"midas:quota:{day}:{kind}:{subject}"


async def get_quota_status(
    request: Request,
    client_id: str | None = Header(default=None, alias="X-MIDAS-Client-ID"),
) -> dict:
    if redis_client is None:
        return {
            "enabled": False,
            "reset_seconds": _seconds_until_utc_midnight(),
            "limits": {
                kind: {"limit": limit, "used": 0, "remaining": limit}
                for kind, limit in DEFAULT_LIMITS.items()
            },
        }

    subject = _subject(request, client_id)
    limits = {}

    for kind, limit in DEFAULT_LIMITS.items():
        used = int(await redis_client.get(_key(kind, subject)) or 0)
        limits[kind] = {
            "limit": limit,
            "used": used,
            "remaining": max(0, limit - used),
        }

    return {
        "enabled": True,
        "reset_seconds": _seconds_until_utc_midnight(),
        "limits": limits,
    }


async def require_quota(
    request: Request,
    kind: QuotaKind,
    client_id: str | None = None,
) -> dict:
    limit = DEFAULT_LIMITS[kind]

    if limit <= 0:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "kind": kind,
                "limit": limit,
                "used": limit,
                "remaining": 0,
                "reset_seconds": _seconds_until_utc_midnight(),
            },
        )

    # Local fallback: do not block dev if Redis is absent.
    # In production, set REDIS_URL. Do not rely on this fallback for public demo.
    if redis_client is None:
        return {
            "kind": kind,
            "limit": limit,
            "used": 0,
            "remaining": limit,
            "reset_seconds": _seconds_until_utc_midnight(),
        }

    subject = _subject(request, client_id)
    key = _key(kind, subject)
    used = int(await redis_client.get(key) or 0)

    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "kind": kind,
                "limit": limit,
                "used": used,
                "remaining": 0,
                "reset_seconds": _seconds_until_utc_midnight(),
            },
        )

    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, _seconds_until_utc_midnight())
    results = await pipe.execute()

    new_used = int(results[0])

    return {
        "kind": kind,
        "limit": limit,
        "used": new_used,
        "remaining": max(0, limit - new_used),
        "reset_seconds": _seconds_until_utc_midnight(),
    }


async def require_upload_quota(request: Request) -> dict:
    return await require_quota(request, "upload", request.headers.get("X-MIDAS-Client-ID"))


async def require_pipeline_quota(request: Request) -> dict:
    return await require_quota(request, "pipeline", request.headers.get("X-MIDAS-Client-ID"))


async def require_reason_quota(request: Request) -> dict:
    return await require_quota(request, "reason", request.headers.get("X-MIDAS-Client-ID"))


async def require_explain_quota(request: Request) -> dict:
    return await require_quota(request, "explain", request.headers.get("X-MIDAS-Client-ID"))


async def require_feedback_quota(request: Request) -> dict:
    return await require_quota(request, "feedback", request.headers.get("X-MIDAS-Client-ID"))