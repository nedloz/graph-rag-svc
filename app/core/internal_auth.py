"""Кто вызвал внутренний API.

Статических токенов нет: вызывающего подтверждает сертификат, а имя из него проставляет
прокси-сайдкар получателя в заголовок X-Caller-Identity (см. infra/mtls в корне main).
Заголовку можно верить только если запрос пришёл с loopback, то есть от прокси в том же
сетевом пространстве; от любого другого адреса он не принимается. Поэтому сервис, до которого
можно достучаться в обход прокси, внутренние маршруты не обслуживает, а не пускает всех.

INTERNAL_AUTH_MODE:
  mtls      (по умолчанию) — верить X-Caller-Identity только от loopback;
  insecure  — без проверки, только для локальной отладки без PKI; сервис громко предупреждает.
"""

import ipaddress
import logging

from fastapi import HTTPException, Request, status

from app.core.config import settings

logger = logging.getLogger(__name__)

CALLER_HEADER = "X-Caller-Identity"
_MODES = ("mtls", "insecure")


def validate_internal_auth_mode() -> None:
    mode = settings.internal_auth_mode
    if mode not in _MODES:
        raise RuntimeError(f"INTERNAL_AUTH_MODE={mode!r}: допустимо {', '.join(_MODES)}")
    if mode == "insecure":
        logger.warning(
            "INTERNAL_AUTH_MODE=insecure: внутренние вызовы не аутентифицируются. "
            "Не для стенда и продакшена."
        )


def _is_loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def caller_identity(headers, peer_host: str | None) -> str | None:
    """Имя вызывающего или None, если запросу доверять нельзя."""
    if settings.internal_auth_mode == "insecure":
        # Имя из X-Service-Name — просто метка трафика (consumer), проверить его нечем.
        return headers.get(CALLER_HEADER) or headers.get("X-Service-Name") or "unauthenticated"
    if not _is_loopback(peer_host):
        return None
    return headers.get(CALLER_HEADER) or None


def verify_internal_http(headers, peer_host: str | None) -> str:
    caller = caller_identity(headers, peer_host)
    if caller is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Internal API: caller not verified")
    return caller


def require_internal_service(request: Request) -> str:
    """FastAPI dependency для внутренних HTTP-маршрутов."""
    return verify_internal_http(request.headers, request.client.host if request.client else None)
