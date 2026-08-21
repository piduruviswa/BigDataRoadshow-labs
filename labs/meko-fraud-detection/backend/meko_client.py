"""
Real MEKO MCP client.

Connects to a live MEKO MCP server over Streamable HTTP using OAuth 2.1
(PKCE + Dynamic Client Registration), per the MCP Authorization spec, via
the official `mcp` Python SDK. Nothing in this file is simulated: it opens
a real network connection, does a real OAuth loopback flow in your
browser, and calls real MCP tools.

Where this is used deliberately, and where it isn't:

  - The synchronous critical path (Agents 1-4 in main.py, budget <= 45ms)
    never calls this client. A network + OAuth round trip cannot fit an
    8-12ms sub-budget, and the whole point of that path is "zero LLM/
    network latency risk on the critical path" - adding a live MCP call
    there would contradict the architecture, not fulfill it.
  - The asynchronous deep-forensic path (Agents 5-6, budget 0.5s-15s) is
    exactly where a real MEKO round trip fits, so that's where demo_app.py
    wires this client in - with a clear, visible fallback to the local
    simulation if MEKO is unreachable or not yet authorized, so a live
    demo never just breaks.

On first use, connect() opens your default browser for the OAuth consent
screen; tokens are cached to ~/.meko/mcp_tokens.json so subsequent runs
don't need to re-authorize.
"""

import asyncio
import contextlib
import json
import os
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx2 as httpx  # the `mcp` SDK's OAuthClientProvider is an httpx2.Auth,
# not a plain httpx.Auth - a plain httpx.AsyncClient rejects it as an invalid
# `auth=` argument at runtime, so the client built here must match.
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

MEKO_MCP_URL = os.environ.get("MEKO_MCP_URL", "https://mcp.mekodev.com/mcp")
TOKEN_CACHE = Path(os.environ.get("MEKO_TOKEN_CACHE", str(Path.home() / ".meko" / "mcp_tokens.json")))
REDIRECT_PORT = int(os.environ.get("MEKO_OAUTH_PORT", "8765"))
REDIRECT_URI = f"http://127.0.0.1:{REDIRECT_PORT}/callback"
CONNECT_TIMEOUT_S = float(os.environ.get("MEKO_CONNECT_TIMEOUT_S", "10"))


class FileTokenStorage:
    """Persists OAuth tokens + the Dynamic-Client-Registration client info
    to a local JSON file, so you only authorize once per laptop."""

    def __init__(self, path: Path = TOKEN_CACHE):
        self.path = path

    def _read(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data))
        with contextlib.suppress(OSError):
            self.path.chmod(0o600)  # tokens are local secrets

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read().get("tokens")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)


class _CallbackServer:
    """Minimal localhost-only HTTP server to catch the OAuth redirect -
    the standard loopback pattern for a native/CLI app per RFC 8252."""

    def __init__(self, port: int):
        self.result: dict[str, str] = {}
        self._event = threading.Event()
        self._server = HTTPServer(("127.0.0.1", port), self._make_handler())
        self._thread = threading.Thread(target=self._server.handle_request, daemon=True)

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib method name
                qs = parse_qs(urlparse(self.path).query)
                outer.result = {
                    "code": qs.get("code", [""])[0],
                    "state": qs.get("state", [""])[0],
                    "error": qs.get("error", [""])[0],
                }
                ok = not outer.result["error"]
                msg = (
                    "Authorization complete — you can close this tab and return to the demo."
                    if ok
                    else f"Authorization failed: {outer.result['error']}"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body style='font-family:sans-serif;padding:2rem'>{msg}</body></html>".encode())
                outer._event.set()

            def log_message(self, *args):  # silence default stderr access log
                pass

        return Handler

    def start(self) -> None:
        self._thread.start()

    async def wait_for_result(self, timeout: float = 180.0) -> dict:
        await asyncio.get_event_loop().run_in_executor(None, self._event.wait, timeout)
        with contextlib.suppress(OSError):
            self._server.server_close()
        return self.result


async def _redirect_handler(auth_url: str) -> None:
    print(f"[MEKO] Opening your browser to authorize:\n  {auth_url}")
    webbrowser.open(auth_url)


async def _callback_handler() -> AuthorizationCodeResult:
    server = _CallbackServer(REDIRECT_PORT)
    server.start()
    result = await server.wait_for_result()
    if not result:
        raise RuntimeError("Timed out waiting for the MEKO OAuth redirect")
    if result.get("error"):
        raise RuntimeError(f"MEKO authorization failed: {result['error']}")
    return AuthorizationCodeResult(code=result["code"], state=result.get("state") or None)


def _build_oauth_provider(server_url: str) -> OAuthClientProvider:
    metadata = OAuthClientMetadata(
        redirect_uris=[REDIRECT_URI],
        client_name="MEKO Fraud Detection Demo",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
    )


@dataclass
class MekoConnection:
    connected: bool
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


class MekoClient:
    """Thin wrapper over an MCP ClientSession to MEKO. Every method here
    makes a real network call - there is no simulated data in this file.
    Callers are responsible for falling back to local logic on failure."""

    def __init__(self, url: str = MEKO_MCP_URL):
        self.url = url
        self._session: ClientSession | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self.tools: dict[str, Any] = {}
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self, timeout: float = CONNECT_TIMEOUT_S) -> MekoConnection:
        """Bounded by asyncio.wait_for rather than only the httpx client's
        own timeout: a stalled connection several layers down (proxy black-
        hole, slow DNS, a half-open TCP handshake) needs one authoritative
        deadline that reliably surfaces as a catchable TimeoutError, not an
        asyncio.CancelledError - which is a BaseException, not an Exception,
        and would otherwise crash the FastAPI startup event instead of
        degrading to the local simulation."""
        await self.close()
        try:
            return await asyncio.wait_for(self._connect_once(), timeout=timeout)
        except TimeoutError:
            self.last_error = f"Timed out after {timeout}s connecting to {self.url}"
        except Exception as exc:  # noqa: BLE001 - surfaced to caller, never swallowed silently
            self.last_error = f"{type(exc).__name__}: {exc}"
        await self.close()
        return MekoConnection(connected=False, error=self.last_error)

    async def _connect_once(self) -> MekoConnection:
        stack = contextlib.AsyncExitStack()
        try:
            oauth = _build_oauth_provider(self.url)
            http_client = httpx.AsyncClient(auth=oauth)
            read, write = await stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()

            self._stack = stack
            self._session = session
            self.tools = {t.name: t for t in listed.tools}
            self.last_error = None
            return MekoConnection(connected=True, tool_names=sorted(self.tools))
        except BaseException:
            await stack.aclose()
            raise

    def find_tool(self, *keywords: str) -> str | None:
        """Best-effort match of a discovered tool by name/description
        keywords. MEKO's real tool names aren't known ahead of connecting -
        this adapts to whatever list_tools() actually returns instead of
        guessing blind. Exact wiring should replace this once the real
        tool names are confirmed against a reachable MEKO server."""
        for name, tool in self.tools.items():
            haystack = f"{name} {tool.description or ''}".lower()
            if all(kw.lower() in haystack for kw in keywords):
                return name
        return None

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if not self._session:
            raise RuntimeError("Not connected to MEKO")
        return await self._session.call_tool(name, arguments)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self.tools = {}
