import json

from fastapi import Header, HTTPException, WebSocket, status

from app.core.config import settings


def get_internal_auth_headers() -> dict[str, str]:
    return {
        settings.internal_service_name_header: settings.service_name,
        settings.internal_auth_header_name: settings.service_token,
    }


def _trusted_tokens() -> dict[str, str]:
    try:
        return json.loads(settings.trusted_service_tokens or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid TRUSTED_SERVICE_TOKENS JSON") from exc


def verify_internal_http_headers(service_name: str | None, service_token: str | None) -> str:
    if not settings.internal_auth_enabled:
        return service_name or settings.service_name
    trusted = _trusted_tokens()
    expected = trusted.get(service_name or "")
    if not expected or expected != (service_token or ""):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal service token")
    return service_name or settings.service_name


def require_internal_service(
    x_service_name: str = Header(..., alias=settings.internal_service_name_header),
    x_service_token: str = Header(..., alias=settings.internal_auth_header_name),
) -> str:
    """FastAPI dependency for HTTP routes — thin adapter over
    verify_internal_http_headers() so route signatures stay declarative."""
    return verify_internal_http_headers(x_service_name, x_service_token)


async def verify_internal_websocket(websocket: WebSocket) -> str | None:
    if not settings.internal_auth_enabled:
        return websocket.headers.get(settings.internal_service_name_header)
    trusted = _trusted_tokens()
    service_name = websocket.headers.get(settings.internal_service_name_header, "")
    service_token = websocket.headers.get(settings.internal_auth_header_name, "")
    expected = trusted.get(service_name)
    if not expected or expected != service_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    return service_name
