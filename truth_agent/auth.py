import hmac
import os
import threading
import logging
import time
from typing import Iterable, Optional
from urllib.parse import urlparse
import anyio
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token as google_id_token
from mcp.shared._httpx_utils import create_mcp_http_client
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
_TOKEN_TTL_SECONDS = 2700

def iam_auth_enabled() -> bool:
    return os.getenv("A2A_USE_IAM_AUTH","").strip().lower() in ("1","true","yes")

def audience_for(url:str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Cannot derive an audience from URL: {url}")
    return f"{parsed.scheme}://{parsed.netloc}"

class GoogleIdTokenAuth(httpx.Auth):

    def __init__(self,audience:str):
        self._audience = audience
        self._token : Optional[str] = None
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def _fetch_token(self) -> str:
        with self._lock:
            now = time.monotonic()
            if self._token and (now-self._fetched_at) < _TOKEN_TTL_SECONDS:
                return self._token
            self._token = google_id_token.fetch_id_token(
                GoogleAuthRequest(), self._audience
            )
            self._fetched_at = now
            return self._token

    def sync_auth_flow(self,request):
        request.headers["Authorization"] = f"Bearer {self._fetch_token()}"
        yield request
        
    async def async_auth_flow(self, request):
        token = await anyio.to_thread.run_sync(self._fetch_token)
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

def a2a_httpx_client(target_url: str, timeout: float = 600.0):
    if not iam_auth_enabled():
        return None
    return httpx.AsyncClient(
        auth=GoogleIdTokenAuth(audience_for(target_url)),
        timeout=timeout,
        follow_redirects=True
    )

def mcp_httpx_client_factory(target_url: str):

    if not iam_auth_enabled():
        return create_mcp_http_client

    id_token_auth = GoogleIdTokenAuth(audience_for(target_url))

    def factory(headers=None, timeout=None, auth=None):
        
        return create_mcp_http_client(
            headers=headers,
            timeout=timeout,
            auth=auth or id_token_auth)

    return factory

####### USER SIDE #####
API_KEY_HEADER = "X-API-Key"
class ApiKeyMiddleWare(BaseHTTPMiddleware):

    def __init__(self,app,token:str,exempt_paths: Iterable[str] = ()):
        super().__init__(app)
        self._expected = token.encode()
        self._exempt = set(exempt_paths)

    async def dispatch(self,request,call_next):
        if request.url.path in self._exempt:
            return await call_next(request)

        presented = request.headers.get(API_KEY_HEADER,"").encode("utf-8","replace")
        if not hmac.compare_digest(presented,self._expected):
            return JSONResponse({"detail":"Unauthorized"},status_code=401)

        return await call_next(request)

def install_api_key_auth(app,exempt_paths: Iterable[str] = ("/health",)) -> bool:
    token = os.getenv("API_BEARER_TOKEN","").strip()
    if not token:
        logger.warning("API_BEARER_TOKEN is not set - the orchestrator API is unauthenticated")
        return False

    app.add_middleware(ApiKeyMiddleWare,token=token,exempt_paths=exempt_paths)
    logger.info(f"API key auth enabled on {API_KEY_HEADER} header")
    return True