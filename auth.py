"""
auth.py — JWT authentication and RBAC permission checking for MCP Server.

Validates agent tokens issued by Auth-API (RS256 JWTs).
Checks permissions before tool execution based on the agent's role.

Flow:
  1. Agent sends request with Authorization: Bearer <jwt>
  2. Middleware validates JWT signature against Auth-API public key
  3. Extracts agent identity (name, role, department, permissions)
  4. Before tool execution, checks if agent has mcp:<server>:<tool> permission
  5. Propagates agent context to backend API calls via X-Agent-* headers

Configuration:
  AUTH_API_JWKS_URL   — JWKS endpoint (default: https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json)
  AUTH_REQUIRE_JWT    — If "true", reject requests without JWT (default: "false" for transition)
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import json as _json

import httpx
import jwt
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ContextVars holding the current caller's identity for the duration of a
# single MCP request. FastMCP tool functions don't get the Request object
# directly, but ContextVars propagate through asyncio.copy_context() across
# every task spawned during the request, so tools can read these to forward
# the caller's JWT to upstream APIs (content-api, auth-api, etc.).
#
# JWTAuthMiddleware sets them; tools read via current_caller_jwt() etc.
_current_jwt: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_caller_jwt", default=""
)
_current_agent_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_caller_agent", default=""
)


def current_caller_jwt() -> str:
    """Return the JWT (raw token string) of the agent that made the
    current MCP request, or empty string if anonymous/no auth.
    """
    return _current_jwt.get()


def current_caller_agent_name() -> str:
    """Return the agent_name claim of the current caller, or empty string."""
    return _current_agent_name.get()

logger = logging.getLogger("mcp-auth")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

AUTH_API_JWKS_URL = os.getenv(
    "AUTH_API_JWKS_URL",
    "https://auth-api.arkturian.com/api/v1/auth/.well-known/jwks.json",
)
AUTH_API_ISSUER = os.getenv("AUTH_API_ISSUER", "auth-api.arkturian.com")
AUTH_REQUIRE_JWT = os.getenv("AUTH_REQUIRE_JWT", "false").lower() == "true"

# Paths that don't require auth
PUBLIC_PATHS = {"/", "/health", "/.well-known/mcp.json", "/docs", "/openapi.json"}


# ──────────────────────────────────────────────────────────────
# Agent context (stored in request.state.agent)
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """Identity and permissions of the authenticated agent."""
    name: str = ""
    agent_type: str = ""
    role: str = ""
    department: str = ""
    tenant: str = ""
    permissions: list[str] = field(default_factory=lambda: ["*"])
    authenticated: bool = False

    def has_permission(self, required: str) -> bool:
        """Check if this agent has a specific permission."""
        return any(_match_permission(p, required) for p in self.permissions)


def _match_permission(held: str, required: str) -> bool:
    """Wildcard permission matching."""
    if held == "*":
        return True
    held_parts = held.split(":")
    required_parts = required.split(":")
    for i, hp in enumerate(held_parts):
        if hp == "*":
            return True
        if i >= len(required_parts):
            return False
        if hp != required_parts[i]:
            return False
    return len(held_parts) == len(required_parts)


# ──────────────────────────────────────────────────────────────
# JWKS public key cache
# ──────────────────────────────────────────────────────────────

_public_key = None
_public_key_fetched_at = 0.0
_PUBLIC_KEY_TTL = 3600  # refresh every hour


async def _get_public_key():
    """Fetch and cache the Auth-API public key from JWKS endpoint."""
    global _public_key, _public_key_fetched_at

    if _public_key and (time.time() - _public_key_fetched_at < _PUBLIC_KEY_TTL):
        return _public_key

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(AUTH_API_JWKS_URL)
            res.raise_for_status()
            jwks = res.json()

        # Extract RSA public key from JWKS
        import base64
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
        from cryptography.hazmat.backends import default_backend

        key_data = jwks["keys"][0]
        n_bytes = base64.urlsafe_b64decode(key_data["n"] + "==")
        e_bytes = base64.urlsafe_b64decode(key_data["e"] + "==")
        n = int.from_bytes(n_bytes, "big")
        e = int.from_bytes(e_bytes, "big")

        _public_key = RSAPublicNumbers(e, n).public_key(default_backend())
        _public_key_fetched_at = time.time()
        logger.info("JWKS public key fetched from %s", AUTH_API_JWKS_URL)
        return _public_key
    except Exception as exc:
        logger.error("Failed to fetch JWKS from %s: %s", AUTH_API_JWKS_URL, exc)
        return _public_key  # return cached key if available


def _decode_jwt(token: str, public_key) -> dict:
    """Decode and validate a JWT token."""
    return jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        issuer=AUTH_API_ISSUER,
        options={"verify_exp": True},
    )


# ──────────────────────────────────────────────────────────────
# MCP path → permission mapping
# ──────────────────────────────────────────────────────────────

def _extract_mcp_server_from_path(path: str) -> str:
    """Extract MCP server name from request path.

    /storage/...  → storage
    /business/... → business
    /cloud/...    → cloud
    """
    parts = path.strip("/").split("/")
    return parts[0] if parts else ""


# ──────────────────────────────────────────────────────────────
# FastAPI Middleware
# ──────────────────────────────────────────────────────────────

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Validate JWT tokens and inject AgentContext into request.state."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            request.state.agent = AgentContext()
            return await call_next(request)

        # GET requests = MCP init/SSE handshake — always allow.
        if request.method == "GET":
            request.state.agent = AgentContext()
            return await call_next(request)

        # For POST requests, read the JSON-RPC method to decide auth level.
        # Only "tools/call" requires JWT — initialize, tools/list, notifications
        # are allowed without auth so Claude Code can connect and discover tools.
        # This prevents Claude Code from triggering its built-in OAuth flow.
        body_bytes = None
        jsonrpc_method = ""
        if request.method == "POST":
            try:
                body_bytes = await request.body()
                request._body = body_bytes  # cache for downstream
                body_json = _json.loads(body_bytes)
                jsonrpc_method = body_json.get("method", "")
            except Exception:
                pass

        # Only enforce auth on tools/call — the actual tool execution
        is_tool_call = jsonrpc_method == "tools/call"
        if not is_tool_call:
            request.state.agent = AgentContext()
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        if not token:
            if AUTH_REQUIRE_JWT:
                return JSONResponse(
                    {"error": "Authorization required for tool calls"},
                    status_code=403,  # 403 not 401 — avoids OAuth trigger
                )
            # No token = anonymous (transition period)
            request.state.agent = AgentContext()
            return await call_next(request)

        # Validate JWT
        try:
            public_key = await _get_public_key()
            if not public_key:
                # Can't validate — allow in transition mode, deny in strict mode
                if AUTH_REQUIRE_JWT:
                    return JSONResponse(
                        {"error": "Auth service unavailable (JWKS fetch failed)"},
                        status_code=503,
                    )
                request.state.agent = AgentContext()
                return await call_next(request)

            claims = _decode_jwt(token, public_key)

            agent = AgentContext(
                name=claims.get("agent_name", claims.get("sub", "")),
                agent_type=claims.get("agent_type", ""),
                role=claims.get("role", ""),
                department=claims.get("department", ""),
                tenant=claims.get("tenant", ""),
                permissions=claims.get("permissions", ["*"]),
                authenticated=True,
            )
            request.state.agent = agent

            # Stash the raw JWT into ContextVars so tool functions further
            # downstream can read it for upstream API forwarding. Reset on
            # response (no token-leak between requests).
            _jwt_token = _current_jwt.set(token)
            _name_token = _current_agent_name.set(agent.name)

            # Permission check: server-level + tool-level
            mcp_server = _extract_mcp_server_from_path(request.url.path)
            if mcp_server:
                # 1. Server-level: does agent have ANY permission on this MCP?
                has_any = any(
                    p == "*" or p == "mcp:*" or p.startswith(f"mcp:{mcp_server}:")
                    for p in agent.permissions
                )
                if not has_any:
                    logger.warning(
                        "RBAC denied (server): agent=%s role=%s mcp:%s",
                        agent.name, agent.role, mcp_server,
                    )
                    return JSONResponse(
                        {
                            "error": f"Permission denied: agent '{agent.name}' (role '{agent.role}') "
                                     f"does not have access to mcp:{mcp_server}",
                            "agent": agent.name,
                            "role": agent.role,
                            "required": f"mcp:{mcp_server}:*",
                        },
                        status_code=403,
                    )

                # 2. Tool-level: check specific tool from the body we already parsed
                if not agent.has_permission(f"mcp:{mcp_server}:*") and body_bytes:
                    try:
                        body_json = _json.loads(body_bytes)
                        tool_name = body_json.get("params", {}).get("name", "")
                        if tool_name:
                            required_perm = f"mcp:{mcp_server}:{tool_name}"
                            if not agent.has_permission(required_perm):
                                logger.warning(
                                    "RBAC denied (tool): agent=%s role=%s perm=%s",
                                    agent.name, agent.role, required_perm,
                                )
                                return JSONResponse(
                                    {
                                        "jsonrpc": "2.0",
                                        "id": body_json.get("id"),
                                        "error": {
                                            "code": -32600,
                                            "message": f"Permission denied: agent '{agent.name}' "
                                                       f"(role '{agent.role}') cannot call {required_perm}",
                                        },
                                    },
                                    status_code=403,
                                )
                    except Exception:
                        pass

        except jwt.ExpiredSignatureError:
            return JSONResponse({"error": "Token expired"}, status_code=401)
        except jwt.InvalidTokenError as e:
            return JSONResponse({"error": f"Invalid token: {e}"}, status_code=401)

        try:
            return await call_next(request)
        finally:
            # Reset ContextVars after request — prevents JWT bleed between calls
            try:
                _current_jwt.reset(_jwt_token)
                _current_agent_name.reset(_name_token)
            except (LookupError, NameError):
                pass


# ──────────────────────────────────────────────────────────────
# Context propagation headers
# ──────────────────────────────────────────────────────────────

def get_agent_headers(request: Request) -> dict[str, str]:
    """Extract agent context from request and return headers for backend API calls.

    Add these headers to _fetch_json calls so backend APIs know who's calling.
    """
    agent: AgentContext = getattr(request.state, "agent", None) or AgentContext()
    if not agent.authenticated:
        return {}
    return {
        "X-Agent-Name": agent.name,
        "X-Agent-Role": agent.role,
        "X-Agent-Department": agent.department,
        "X-Agent-Tenant": agent.tenant or "",
    }
