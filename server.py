#!/usr/bin/env python3
"""
Arkturian MCP Server

Exposes MCP endpoint groups over HTTP/SSE with per-tenant isolation:
  • /storage        – Arkturian tenant Storage & Knowledge Graph tools
  • /oneal          – O'Neal product catalogue API
  • /oneal-storage  – O'Neal tenant Storage & Knowledge Graph tools (615 products)
  • /artrack        – Artrack GPS tracking & route management API
  • /codepilot      – Human-in-the-loop tools (Telegram notifications & questions)
  • /ai             – AI text, vision, and image generation tools
  • /content        – Content management API for posts, media, annotations, and blocks
  • /tree           – Collaborative tree editing with node-level CRUD and real-time sync
  • /business       – Business management: Honorarnoten, Rechnungen, Kunden, Transaktionen
  • /comm           – Unified communication: Email, Telegram, Interventions
  • /review         – AI-powered multi-perspective review orchestrator
"""

from __future__ import annotations

import json
import logging
import os
import time
import base64
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

STORAGE_API_BASE = os.getenv("ARKTURIAN_API_BASE", "https://api-storage.arkturian.com")
STORAGE_API_KEY = os.getenv("ARKTURIAN_API_KEY", "")

ONEAL_API_BASE = os.getenv("ONEAL_API_BASE", "http://127.0.0.1:8014")
ONEAL_API_KEY = os.getenv("ONEAL_API_KEY", "oneal_demo_token")

# O'Neal Storage API (same base URL as Arkturian, different API key for tenant isolation)
ONEAL_STORAGE_API_KEY = os.getenv("ONEAL_STORAGE_API_KEY", "")

# Artrack API
ARTRACK_API_BASE = os.getenv("ARTRACK_API_BASE", "https://api-artrack.arkturian.com")
ARTRACK_API_KEY = os.getenv("ARTRACK_API_KEY", "")

# Content API
CONTENT_API_BASE = os.getenv("CONTENT_API_BASE", "https://content-api.arkturian.com")
# Public-facing URL for share links / Telegram / browser downloads.
# Falls back to CONTENT_API_BASE when not explicitly set. Set this when
# CONTENT_API_BASE points at an internal address (e.g. http://127.0.0.1:8015).
CONTENT_API_PUBLIC_BASE = os.getenv("CONTENT_API_PUBLIC_BASE", CONTENT_API_BASE)
CONTENT_API_KEY = os.getenv("CONTENT_API_KEY", "").strip()

# Tree API
TREE_API_BASE = os.getenv("TREE_API_BASE", "https://tree-api.arkturian.com")

# CodePilot API (for creating change requests)
CODEPILOT_API_BASE = os.getenv("CODEPILOT_API_BASE", "http://localhost:8201")
CODEPILOT_API_TOKEN = os.getenv("CODEPILOT_API_TOKEN", "")

# AI API
AI_API_BASE = os.getenv("AI_API_BASE", "https://api-ai.arkturian.com")
AI_API_KEY = os.getenv("AI_API_KEY", "")

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8080"))
HTTP_TIMEOUT = float(os.getenv("MCP_HTTP_TIMEOUT", "30.0"))

# Optional MCP filter — defined here early so validation can use it
_mcp_servers_env_early = os.getenv("MCP_SERVERS", "").strip()
_enabled_early: Optional[set] = (
    {s.strip().lower() for s in _mcp_servers_env_early.split(",") if s.strip()}
    if _mcp_servers_env_early
    else None
)

def _requires(name: str) -> bool:
    """Check if an MCP is enabled (or all are enabled if no filter)."""
    return _enabled_early is None or name in _enabled_early

# Only enforce keys for enabled MCPs
if _requires("storage") and not STORAGE_API_KEY:
    raise RuntimeError("ARKTURIAN_API_KEY environment variable must be set.")
if _requires("oneal-storage") and not ONEAL_STORAGE_API_KEY:
    raise RuntimeError("ONEAL_STORAGE_API_KEY environment variable must be set.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("arkturian-mcp")

STORAGE_PATH = "/storage"
ONEAL_PATH = "/oneal"
ONEAL_STORAGE_PATH = "/oneal-storage"
ARTRACK_PATH = "/artrack"
CODEPILOT_PATH = "/codepilot"
AI_PATH = "/ai"
CONTENT_PATH = "/content"
TREE_PATH = "/tree"
TAROT_PATH = "/tarot"
BUSINESS_PATH = "/business"
COMM_PATH = "/comm"
KNOWLEDGE_PATH = "/knowledge"
STORY_PATH = "/story"

# Tools API (for Tarot)
TOOLS_API_BASE = os.getenv("TOOLS_API_BASE", "https://tools-api.arkturian.com")

# Business API
BUSINESS_API_BASE = os.getenv("BUSINESS_API_BASE", "https://business-api.arkturian.com")
BUSINESS_API_KEY = os.getenv("BUSINESS_API_KEY", "")

# Comm API
COMM_API_BASE = os.getenv("COMM_API_BASE", "https://comm-api.arkturian.com")
COMM_API_KEY = os.getenv("COMM_API_KEY", "")

# Story API
STORY_API_BASE = os.getenv("STORY_API_BASE", "http://localhost:8070")
STORY_API_KEY = os.getenv("STORY_API_KEY", "story_ark_secret_2025")

# Knowledge API
KNOWLEDGE_API_BASE = os.getenv("KNOWLEDGE_API_BASE", "https://knowledge-api.arkturian.com")

# Review API
REVIEW_API_BASE = os.getenv("REVIEW_API_BASE", "https://review-api.arkturian.com")
REVIEW_API_KEY = os.getenv("REVIEW_API_KEY", "")
REVIEW_PATH = "/review"

# Cloud API (inter-agent communication)
CLOUD_API_BASE = os.getenv("CLOUD_API_BASE", "http://localhost:8070")

# Conversation API — cross-channel conversation tracking + agent dispatch
CONVERSATION_API_BASE = os.getenv(
    "CONVERSATION_API_BASE",
    "https://conversation-api.arkserver.arkturian.com",
)
CONVERSATION_API_KEY = os.getenv("CONVERSATION_API_KEY", "").strip()
CLOUD_PATH = "/cloud"
CONVERSATION_PATH = "/conversation"
# Shared secret for cross-node IACP role addressing. Must match IACP_TOKEN
# env on every cloud-api in the federation. Used by send_to_role().
IACP_TOKEN = os.getenv("IACP_TOKEN", "").strip()

# --------------------------------------------------------------------------- #
# MCP server filtering — selective deployment
# --------------------------------------------------------------------------- #
# Set MCP_SERVERS env var to a comma-separated list of MCP names to enable
# only those endpoints. Empty/unset = mount all (default behavior).
# Example: MCP_SERVERS=content  → only content endpoint is mounted
_mcp_servers_env = os.getenv("MCP_SERVERS", "").strip()
ENABLED_MCPS: Optional[set] = (
    {s.strip().lower() for s in _mcp_servers_env.split(",") if s.strip()}
    if _mcp_servers_env
    else None
)


def mount_mcp(name: str, path: str, mcp_app) -> bool:
    """Mount an MCP server only if enabled by ENABLED_MCPS filter.

    Args:
        name: Logical MCP name (lowercase, e.g. "content", "storage")
        path: URL path to mount at
        mcp_app: The streamable_http_app() instance

    Returns:
        True if mounted, False if filtered out.
    """
    if ENABLED_MCPS is not None and name.lower() not in ENABLED_MCPS:
        print(f"⊘ Skipping MCP mount: {name} (not in MCP_SERVERS={','.join(sorted(ENABLED_MCPS))})")
        return False
    app.mount(path, mcp_app)
    print(f"✓ Mounted MCP: {name} at {path}")
    return True

# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


async def _fetch_json(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Any:
    async with httpx.AsyncClient(timeout=timeout or HTTP_TIMEOUT) as client:
        try:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params or {})
            elif method == "POST":
                response = await client.post(url, headers=headers, params=params or {}, json=json_body or {})
            elif method == "PUT":
                response = await client.put(url, headers=headers, params=params or {}, json=json_body or {})
            elif method == "PATCH":
                response = await client.patch(url, headers=headers, params=params or {}, json=json_body or {})
            elif method == "DELETE":
                response = await client.delete(url, headers=headers, params=params or {})
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            # Surface the upstream response body to the caller — the default
            # HTTPStatusError.__str__ only carries the status line, which
            # strips away structured FastAPI detail payloads. Agents should
            # see the actual server error message ("URL could not be
            # fetched", "document too large", etc.) not a bare 500.
            body_text = (exc.response.text or "").strip()
            detail = body_text
            try:
                body_json = exc.response.json()
                if isinstance(body_json, dict) and "detail" in body_json:
                    detail = body_json["detail"]
                    if not isinstance(detail, str):
                        detail = json.dumps(detail)
            except Exception:
                pass
            logger.error("HTTP error %s %s: %s", method, url, body_text)
            raise RuntimeError(
                f"Upstream {exc.response.status_code} from {method} {url}: {detail or 'no detail'}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Request to %s failed: %s", url, exc)
            raise


# ──────────────────────────────────────────────────────────────
# Caller auth-header forwarding
# ──────────────────────────────────────────────────────────────
# JWTAuthMiddleware (auth.py) stashes the caller's raw JWT into a ContextVar
# at request time. Tool functions don't get the Request object directly, but
# ContextVars propagate through asyncio.copy_context() across every task
# spawned during the request, so we can read the caller's JWT here and
# forward it to upstream APIs. This way content-api etc. see the actual
# agent identity (with their tenant + permissions) instead of treating
# every MCP call as anonymous.
def _caller_auth_headers(api_key_fallback: str = "") -> Dict[str, str]:
    """Return Authorization headers for upstream API calls.

    Priority:
      1. Caller's JWT (from ContextVar set by JWTAuthMiddleware) → forward as
         `Authorization: Bearer <jwt>` so upstream knows the agent identity.
      2. Static API-key fallback if provided → for backwards-compat or when
         the MCP is called anonymously and we still want some auth.
      3. Empty (anonymous) → upstream applies its anon-policy.
    """
    from auth import current_caller_jwt
    jwt_str = current_caller_jwt()
    if jwt_str:
        return {"Authorization": f"Bearer {jwt_str}"}
    if api_key_fallback:
        return {"X-API-KEY": api_key_fallback}
    return {}


async def call_storage_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    # storage-api expects X-API-KEY — caller's JWT not applicable here.
    return await _fetch_json(
        method,
        f"{STORAGE_API_BASE}{endpoint}",
        headers={"X-API-KEY": STORAGE_API_KEY},
        params=params,
        json_body=json_body,
    )


async def call_storage_upload(
    file_bytes: bytes,
    filename: str,
    *,
    form_fields: Optional[Dict[str, str]] = None,
) -> Any:
    """Upload a file to Storage API via multipart form-data."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            files = {"file": (filename, file_bytes)}
            data = form_fields or {}
            response = await client.post(
                f"{STORAGE_API_BASE}/storage/upload",
                headers={"X-API-KEY": STORAGE_API_KEY},
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Upload error: %s", exc.response.text)
            raise
        except httpx.HTTPError as exc:
            logger.error("Upload request failed: %s", exc)
            raise


async def call_oneal_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    return await _fetch_json(
        method,
        f"{ONEAL_API_BASE}{endpoint}",
        headers={"X-API-Key": ONEAL_API_KEY},
        params=params,
    )


async def call_oneal_storage_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Storage API with O'Neal tenant credentials for tenant isolation."""
    return await _fetch_json(
        method,
        f"{STORAGE_API_BASE}{endpoint}",
        headers={"X-API-KEY": ONEAL_STORAGE_API_KEY},
        params=params,
        json_body=json_body,
    )


async def call_artrack_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Artrack API for GPS tracking and route management."""
    return await _fetch_json(
        method,
        f"{ARTRACK_API_BASE}{endpoint}",
        headers={"X-API-KEY": ARTRACK_API_KEY},
        params=params,
        json_body=json_body,
    )


async def call_ai_api(
    method: str,
    endpoint: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call AI API (text/image/audio) with API key if provided."""
    headers = {"X-API-Key": AI_API_KEY} if AI_API_KEY else {}
    return await _fetch_json(
        method,
        f"{AI_API_BASE}{endpoint}",
        headers=headers,
        json_body=json_body,
    )


async def call_tools_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Tools API (no auth required for public endpoints)."""
    return await _fetch_json(
        method,
        f"{TOOLS_API_BASE}{endpoint}",
        headers={},
        params=params,
        json_body=json_body,
    )


async def call_knowledge_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Knowledge API (no auth required)."""
    return await _fetch_json(
        method,
        f"{KNOWLEDGE_API_BASE}{endpoint}",
        headers={},
        params=params,
        json_body=json_body,
    )


async def call_review_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Review API with API key auth."""
    return await _fetch_json(
        method,
        f"{REVIEW_API_BASE}{endpoint}",
        headers={"X-API-Key": REVIEW_API_KEY} if REVIEW_API_KEY else {},
        params=params,
        json_body=json_body,
        timeout=180.0,  # Reviews can take a while
    )


async def call_cloud_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Cloud API for inter-agent communication.

    Forwards the caller's JWT so cloud-api's RBAC filters (allowed_agents,
    tenant scoping) apply to the right identity.
    """
    return await _fetch_json(
        method,
        f"{CLOUD_API_BASE}{endpoint}",
        headers=_caller_auth_headers(),
        params=params,
        json_body=json_body,
        timeout=180.0,
    )


async def call_codepilot_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call CodePilot API using Bearer token auth."""
    if not CODEPILOT_API_TOKEN:
        raise RuntimeError("CODEPILOT_API_TOKEN not configured")

    headers = {"Authorization": f"Bearer {CODEPILOT_API_TOKEN}"}
    return await _fetch_json(
        method,
        f"{CODEPILOT_API_BASE}{endpoint}",
        headers=headers,
        params=params,
        json_body=json_body,
    )


async def call_conversation_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Conversation API for cross-channel thread tracking + agent dispatch.

    Sends X-API-KEY header if CONVERSATION_API_KEY env var is set.
    """
    headers: Dict[str, str] = {}
    if CONVERSATION_API_KEY:
        headers["X-API-KEY"] = CONVERSATION_API_KEY
    return await _fetch_json(
        method,
        f"{CONVERSATION_API_BASE}{endpoint}",
        headers=headers,
        params=params,
        json_body=json_body,
    )


async def call_content_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Content API for posts, media, annotations, and blocks.

    Forwards the caller's JWT (so content-api sees their real agent identity
    + tenant) when available; falls back to CONTENT_API_KEY if set.
    """
    return await _fetch_json(
        method,
        f"{CONTENT_API_BASE}{endpoint}",
        headers=_caller_auth_headers(api_key_fallback=CONTENT_API_KEY),
        params=params,
        json_body=json_body,
    )


async def call_tree_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Tree API for projects, nodes, and tree operations.

    Forwards the caller's JWT for tenant-aware filtering when supported.
    """
    return await _fetch_json(
        method,
        f"{TREE_API_BASE}{endpoint}",
        headers=_caller_auth_headers(),
        params=params,
        json_body=json_body,
    )


async def call_business_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Call Business API for invoicing, clients, transactions, and documents."""
    effective_key = api_key or BUSINESS_API_KEY
    return await _fetch_json(
        method,
        f"{BUSINESS_API_BASE}{endpoint}",
        headers={"X-API-Key": effective_key},
        params=params,
        json_body=json_body,
    )


async def call_comm_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Any:
    """Call Comm API for email, telegram, and unified messaging."""
    return await _fetch_json(
        method,
        f"{COMM_API_BASE}{endpoint}",
        headers={"X-API-Key": COMM_API_KEY},
        params=params,
        json_body=json_body,
        timeout=timeout,
    )


async def call_story_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Story Architect API for cinematic storyboard management."""
    return await _fetch_json(
        method,
        f"{STORY_API_BASE}{endpoint}",
        headers={"X-API-KEY": STORY_API_KEY},
        params=params,
        json_body=json_body,
    )


def _clean_params(**kwargs: Any) -> Dict[str, Any]:
    """Drop keys whose values are None."""
    return {key: value for key, value in kwargs.items() if value is not None}


# --------------------------------------------------------------------------- #
# MCP servers
# --------------------------------------------------------------------------- #

# Storage MCP ---------------------------------------------------------------
storage_mcp = FastMCP(
    name="arkturian-storage",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@storage_mcp.tool(
    name="assets_list",
    description="""List storage objects with filters and full-text search.

    Parameters:
    - mine: Only own objects (default true, admin can set false for all)
    - context: Filter by context tag
    - collection_id: Exact collection match
    - collection_like: Partial collection match (case-insensitive)
    - name: Filename contains (case-insensitive)
    - ext: File extension filter (e.g. 'mp4', 'png')
    - link_id: Filter by link_id (supports multi-value with semicolons)
    - search: Full-text search across filename, title, description, ai_title, ai_tags
    - mime_type: MIME type filter ('video' matches video/*, 'image/png' matches exactly)
    - has_hls: Filter by HLS availability (true = only transcoded videos)
    - min_id: Only objects with id >= min_id
    - max_id: Only objects with id <= max_id
    - sort: Sort field ('created_at', 'id', 'filename', 'file_size'). Prefix '-' for ascending.
    - offset: Pagination offset (default 0)
    - limit: Max results (default 100, max 5000)
    """,
)
async def storage_assets_list(
    mine: Optional[bool] = True,
    context: Optional[str] = None,
    collection_id: Optional[str] = None,
    collection_like: Optional[str] = None,
    name: Optional[str] = None,
    ext: Optional[str] = None,
    link_id: Optional[str] = None,
    search: Optional[str] = None,
    mime_type: Optional[str] = None,
    has_hls: Optional[bool] = None,
    min_id: Optional[int] = None,
    max_id: Optional[int] = None,
    sort: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict[str, Any]:
    params = _clean_params(
        mine=mine,
        context=context,
        collection_id=collection_id,
        collection_like=collection_like,
        name=name,
        ext=ext,
        link_id=link_id,
        search=search,
        mime_type=mime_type,
        has_hls=has_hls,
        min_id=min_id,
        max_id=max_id,
        sort=sort,
        offset=offset,
        limit=limit,
    )
    return await call_storage_api("GET", "/storage/list", params=params)


@storage_mcp.tool(
    name="assets_get",
    description="""Get complete storage object with AI-analyzed metadata.
    
    Returns comprehensive data including:
    - Basic info: id, title, file_url, mime_type, dimensions
    - AI fields: ai_title, ai_tags, ai_safety_rating, ai_collections
    - ai_context_metadata: Full structured analysis (product_analysis, visual_analysis, layout_intelligence, semantic_properties)
    
    The ai_context_metadata contains detailed vision AI analysis powering semantic search and recommendations.
    """,
)
async def storage_assets_get(id: int) -> Dict[str, Any]:  # noqa: A002
    return await call_storage_api("GET", f"/storage/objects/{id}")


@storage_mcp.tool(
    name="assets_similar",
    description="""Find semantically similar objects using Knowledge Graph embeddings.
    
    Uses 3072-dim vectors in Chroma DB to find visual and semantic matches.
    Distance scores: 0.0-0.3 (very similar), 0.3-0.7 (related), 0.7+ (different).
    Results ranked by cosine distance.
    """,
)
async def storage_assets_similar(id: int, limit: int = 10) -> Dict[str, Any]:  # noqa: A002
    return await call_storage_api("GET", f"/storage/similar/{id}", params={"limit": limit})


@storage_mcp.tool(
    name="media_preview",
    description="Return a preview URL for a media asset.",
)
async def storage_media_preview(
    id: int,  # noqa: A002
    variant: Optional[str] = None,
    display_for: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Dict[str, Any]:
    options = _clean_params(
        variant=variant,
        display_for=display_for,
        width=width,
        height=height,
        format=format,
        quality=quality,
    )
    url = httpx.URL(f"{STORAGE_API_BASE}/storage/media/{id}").copy_with(params=options)
    return {"url": str(url), "parameters": options}


@storage_mcp.tool(
    name="media_as_data_url",
    description="""DISABLED — base64 image loading causes context overflow (215KB+ payloads).
    Use media_preview instead to get the direct URL, then use Read tool to view the image.
    Example: media_preview(id=123) → url, then Read(url) to see the image.""",
)
async def storage_media_as_data_url(
    id: int,  # noqa: A002
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "error": "media_as_data_url is disabled — base64 payloads exceed context limits. "
                 f"Use media_preview(id={id}) to get the URL, then Read the URL to view the image.",
        "alternative": f"{STORAGE_API_BASE}/storage/media/{id}",
    }


@storage_mcp.tool(
    name="kg_embed",
    description="Create or refresh embeddings for a storage object.",
)
async def storage_kg_embed(id: int) -> Dict[str, Any]:  # noqa: A002
    return await call_storage_api("POST", f"/storage/objects/{id}/embed")


@storage_mcp.tool(
    name="kg_stats",
    description="""Get Knowledge Graph statistics and health metrics.
    
    Returns total embeddings, breakdown by tenant, vector dimensions (3072), and system status.
    Use this to monitor embedding coverage and verify multi-tenancy isolation.
    """,
)
async def storage_kg_stats() -> Dict[str, Any]:
    return await call_storage_api("GET", "/storage/kg/stats")


@storage_mcp.tool(
    name="assets_refs",
    description="Resolve asset variant references.",
)
async def storage_assets_refs(
    link_id: Optional[str] = None,
    collection_id: Optional[str] = None,
    object_id: Optional[int] = None,
    role: Optional[str] = None,
    mine: Optional[bool] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    params = _clean_params(
        link_id=link_id,
        collection_id=collection_id,
        object_id=object_id,
        role=role,
        mine=mine,
        limit=limit,
    )
    return await call_storage_api("GET", "/storage/asset-refs", params=params)


@storage_mcp.tool(
    name="kg_health",
    description="Fetch knowledge graph health diagnostics.",
)
async def storage_kg_health() -> Dict[str, Any]:
    return await call_storage_api("GET", "/storage/kg/health")


@storage_mcp.tool(
    name="kg_search",
    description="""Semantic text search across all storage objects.
    
    Natural language search powered by OpenAI embeddings. Understands synonyms, context, and visual concepts.
    Example: "red motocross gloves for mountain biking" finds relevant products even without exact keyword matches.
    """,
)
async def storage_kg_search(
    query: str,
    limit: int = 10,
    collection_like: Optional[str] = None,
    mine: Optional[bool] = None,
) -> Dict[str, Any]:
    params = _clean_params(
        query=query,
        limit=limit,
        collection_like=collection_like,
        mine=mine,
    )
    return await call_storage_api("GET", "/storage/kg/search", params=params)




@storage_mcp.tool(
    name="assets_get_embedding_text",
    description="""Get the embedding text for a storage object.
    
    The embedding text is a 400-1000 character description that combines all AI metadata into searchable text.
    It is converted to a 3072-dimensional vector for semantic search in the Knowledge Graph.
    
    Returns:
    - object_id: Storage object ID
    - title: Object title
    - embedding_text: Full embedding description
    - searchable_fields: Fields included in search index
    - char_count: Character count
    """,
)
async def storage_assets_get_embedding_text(id: int) -> Dict[str, Any]:  # noqa: A002
    return await call_storage_api("GET", f"/storage/objects/{id}/embedding-text")


@storage_mcp.tool(
    name="assets_update_embedding_text",
    description="""Update embedding text and regenerate Knowledge Graph vector.
    
    Manually refine the searchable description. The system will:
    1. Save new text to ai_context_metadata
    2. Generate new 3072-dim vector (OpenAI text-embedding-3-large)
    3. Update Chroma Vector DB automatically
    
    Use this to improve semantic search results by adding domain-specific keywords or refining descriptions.
    """,
)
async def storage_assets_update_embedding_text(
    id: int,  # noqa: A002
    embedding_text: str
) -> Dict[str, Any]:
    return await call_storage_api(
        "PUT",
        f"/storage/objects/{id}/embedding-text",
        json_body={"embedding_text": embedding_text}
    )


@storage_mcp.tool(
    name="assets_upload",
    description="""Upload a file to Storage via base64-encoded content.

    Accepts file content as a base64 string and uploads it to Storage API.
    The file is stored, and optionally analyzed by AI (vision, safety, embedding).

    Parameters:
    - file_base64: Base64-encoded file content (required)
    - filename: Original filename with extension, e.g. 'photo.jpg' (required)
    - context: Metadata context tag (e.g. 'product image', 'documentation')
    - collection_id: Group into a collection (e.g. 'helmets_2026')
    - link_id: Link related files together
    - is_public: Make publicly accessible (default false)
    - ai_mode: AI analysis level — 'none', 'safety' (default), 'vision', 'full'

    Returns the created storage object with id, file_url, thumbnail_url, ai_title, etc.

    Example: Upload a small file
      assets_upload(file_base64="iVBORw0KGgo...", filename="logo.png", context="branding")
    """,
)
async def storage_assets_upload(
    file_base64: str,
    filename: str,
    context: Optional[str] = None,
    collection_id: Optional[str] = None,
    link_id: Optional[str] = None,
    is_public: bool = False,
    ai_mode: str = "safety",
) -> Dict[str, Any]:
    file_bytes = base64.b64decode(file_base64)
    form_fields: Dict[str, str] = {"ai_mode": ai_mode, "is_public": str(is_public).lower()}
    if context:
        form_fields["context"] = context
    if collection_id:
        form_fields["collection_id"] = collection_id
    if link_id:
        form_fields["link_id"] = link_id
    return await call_storage_upload(file_bytes, filename, form_fields=form_fields)


@storage_mcp.tool(
    name="assets_fetch",
    description="""Import a file from a URL into Storage.

    Downloads the file from the given URL and stores it in Storage API.
    Useful for importing images/documents from the web without manual download.

    Parameters:
    - url: Public URL to fetch the file from (required)
    - context: Metadata context tag
    - collection_id: Group into a collection
    - link_id: Link related files together
    - is_public: Make publicly accessible (default false)
    - filename: Override filename (auto-derived from URL if omitted)
    - analyze: Trigger AI analysis (default true)

    Returns the created storage object with id, file_url, thumbnail_url, etc.

    Example: Import an image from a URL
      assets_fetch(url="https://example.com/photo.jpg", context="imported", collection_id="web_imports")
    """,
)
async def storage_assets_fetch(
    url: str,
    context: Optional[str] = None,
    collection_id: Optional[str] = None,
    link_id: Optional[str] = None,
    is_public: bool = False,
    filename: Optional[str] = None,
    analyze: bool = True,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"url": url, "is_public": is_public, "analyze": analyze}
    if context:
        body["context"] = context
    if collection_id:
        body["collection_id"] = collection_id
    if link_id:
        body["link_id"] = link_id
    if filename:
        body["filename"] = filename
    return await call_storage_api("POST", "/storage/fetch", json_body=body)


# O'Neal MCP ---------------------------------------------------------------
oneal_mcp = FastMCP(
    name="oneal-products",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@oneal_mcp.tool(
    name="products_list",
    description="""List O'Neal products with Storage media URLs.

    Returns paginated list of O'Neal products enriched with storage.media_url fields.
    Each product includes transformable media URLs with automatic caching.

    Filter parameters (all optional, AND-combined):
    - search: whitespace-separated tokens; each must substring-match name OR product_code (AND across tokens)
    - category: category slug (e.g. "helmets-mx", "protection-mtb"). Use facets_list to discover slugs.
    - price_min / price_max: price range in EUR
    - target_group: "Erwachsene" or "Jugendliche" (use this for "Kinder"/"Youth" queries)
    - body_part: "Kopf", "Hand", "Knie", "Ellbogen", etc.
    - product_type: "Helm", "Jersey", "Pant", etc.
    - product_function: "Schutz", "Bekleidung", etc.
    - sport: "MX" or "MTB"
    - color: substring match on color name (English or German), e.g. "blue"/"blau"
    - include_discontinued: false to exclude discontinued products (default true)
    - sort: one of "name", "price", "code", "updated"
    - order: "asc" (default) or "desc"
    - limit / offset: pagination

    Strategy for natural-language queries:
    1. Map user terms via facets_list (slugs/colors/sizes) FIRST
    2. Combine multiple structured filters in ONE call rather than guessing free-text
    3. Use products_get only for individual deep-dives, not for bulk filtering

    Use storage.media_url + transformation parameters to get optimized images.
    See products_get for full transformation parameter documentation.
    """,
)
async def oneal_products_list(
    search: Optional[str] = None,
    category: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    target_group: Optional[str] = None,
    body_part: Optional[str] = None,
    product_type: Optional[str] = None,
    product_function: Optional[str] = None,
    sport: Optional[str] = None,
    color: Optional[str] = None,
    include_discontinued: Optional[bool] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    params = _clean_params(
        search=search,
        category=category,
        price_min=price_min,
        price_max=price_max,
        target_group=target_group,
        body_part=body_part,
        product_type=product_type,
        product_function=product_function,
        sport=sport,
        color=color,
        include_discontinued=include_discontinued,
        sort=sort,
        order=order,
        limit=limit,
        offset=offset,
    )
    return await call_oneal_api("GET", "/v1/products", params=params)


@oneal_mcp.tool(
    name="products_get",
    description="""Fetch O'Neal product with Storage media URLs.

    Returns O'Neal product data enriched with storage.media_url field.
    The media URLs support dynamic image transformation with caching:

    Each product includes:
    - All original O'Neal product data (name, price, category, etc.)
    - storage.id: Storage object ID
    - storage.media_url: Transformable media URL (https://api-storage.arkturian.com/storage/media/{id})
    - storage.thumbnail_url: Pre-generated thumbnail
    - storage.ai_title: AI-generated product title
    - storage.ai_tags: AI-extracted keywords
    - storage.transform_hints: Example transformation parameters

    Image transformation parameters (append to media_url):
    - ?width=400 - Resize to 400px width
    - ?height=300 - Resize to 300px height
    - ?format=webp - Convert to WebP (also jpg, png)
    - ?quality=80 - Set quality (1-100)
    - ?variant=thumbnail - Use preset thumbnail size
    - ?variant=medium - Use preset medium size (1920px)

    Examples:
    - storage.media_url + "?width=400&format=webp&quality=80" → 400px WebP @ 80% quality
    - storage.media_url + "?variant=thumbnail" → Thumbnail preset

    All transformations are automatically cached for fast subsequent access.
    """,
)
async def oneal_products_get(product_id: str) -> Dict[str, Any]:
    return await call_oneal_api("GET", f"/v1/products/{product_id}")


@oneal_mcp.tool(
    name="facets_list",
    description="""List product facets to discover filterable values.

    Returns:
    - categories: list of {slug, name, count} — use slug as `category` param in products_list
    - colors: list of {name, count} — note: products_list does NOT yet support color filter,
      but the values here are useful for building search terms or for products_get inspection
    - sizes: list of {name, count}
    - price_range: {min, max, currency}

    Always call this first when the user query mentions a category, color, or price range
    so you use the correct slugs/values that actually exist in the catalog.
    """,
)
async def oneal_facets_list() -> Dict[str, Any]:
    return await call_oneal_api("GET", "/v1/facets")


@oneal_mcp.tool(
    name="service_ping",
    description="Health check for the O'Neal Product API.",
)
async def oneal_service_ping() -> Dict[str, Any]:
    return await call_oneal_api("GET", "/v1/ping")


# O'Neal Storage MCP --------------------------------------------------------
oneal_storage_mcp = FastMCP(
    name="oneal-storage",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@oneal_storage_mcp.tool(
    name="assets_list",
    description="List O'Neal tenant storage objects with optional filters.",
)
async def oneal_storage_assets_list(
    mine: Optional[bool] = True,
    context: Optional[str] = None,
    collection_id: Optional[str] = None,
    collection_like: Optional[str] = None,
    name: Optional[str] = None,
    ext: Optional[str] = None,
    link_id: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    params = _clean_params(
        mine=mine,
        context=context,
        collection_id=collection_id,
        collection_like=collection_like,
        name=name,
        ext=ext,
        link_id=link_id,
        limit=limit,
    )
    return await call_oneal_storage_api("GET", "/storage/list", params=params)


@oneal_storage_mcp.tool(
    name="assets_get",
    description="""Get complete O'Neal storage object with AI-analyzed metadata.

    Returns comprehensive data including:
    - Basic info: id, title, file_url, mime_type, dimensions
    - AI fields: ai_title, ai_tags, ai_safety_rating, ai_collections
    - ai_context_metadata: Full structured analysis (product_analysis, visual_analysis, layout_intelligence, semantic_properties)

    The ai_context_metadata contains detailed vision AI analysis powering semantic search and recommendations.
    """,
)
async def oneal_storage_assets_get(id: int) -> Dict[str, Any]:  # noqa: A002
    return await call_oneal_storage_api("GET", f"/storage/objects/{id}")


@oneal_storage_mcp.tool(
    name="assets_similar",
    description="""Find semantically similar objects in O'Neal tenant using Knowledge Graph embeddings.

    Uses 3072-dim vectors in Chroma DB to find visual and semantic matches within O'Neal tenant.
    Distance scores: 0.0-0.3 (very similar), 0.3-0.7 (related), 0.7+ (different).
    Results ranked by cosine distance.
    """,
)
async def oneal_storage_assets_similar(id: int, limit: int = 10) -> Dict[str, Any]:  # noqa: A002
    return await call_oneal_storage_api("GET", f"/storage/similar/{id}", params={"limit": limit})


@oneal_storage_mcp.tool(
    name="media_preview",
    description="Return a preview URL for an O'Neal media asset.",
)
async def oneal_storage_media_preview(
    id: int,  # noqa: A002
    variant: Optional[str] = None,
    display_for: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Dict[str, Any]:
    options = _clean_params(
        variant=variant,
        display_for=display_for,
        width=width,
        height=height,
        format=format,
        quality=quality,
    )
    url = httpx.URL(f"{STORAGE_API_BASE}/storage/media/{id}").copy_with(params=options)
    return {"url": str(url), "parameters": options}


@oneal_storage_mcp.tool(
    name="media_as_data_url",
    description="""DISABLED — base64 image loading causes context overflow.
    Use media_preview instead to get the direct URL, then use Read tool to view the image.""",
)
async def oneal_storage_media_as_data_url(
    id: int,  # noqa: A002
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "error": "media_as_data_url is disabled — base64 payloads exceed context limits. "
                 f"Use media_preview(id={id}) to get the URL, then Read the URL to view the image.",
    }


@oneal_storage_mcp.tool(
    name="kg_embed",
    description="Create or refresh embeddings for an O'Neal storage object.",
)
async def oneal_storage_kg_embed(id: int) -> Dict[str, Any]:  # noqa: A002
    return await call_oneal_storage_api("POST", f"/storage/objects/{id}/embed")


@oneal_storage_mcp.tool(
    name="kg_stats",
    description="""Get Knowledge Graph statistics and health metrics for O'Neal tenant.

    Returns total embeddings, breakdown by tenant, vector dimensions (3072), and system status.
    Use this to monitor O'Neal embedding coverage and verify multi-tenancy isolation.
    """,
)
async def oneal_storage_kg_stats() -> Dict[str, Any]:
    return await call_oneal_storage_api("GET", "/storage/kg/stats")


@oneal_storage_mcp.tool(
    name="assets_refs",
    description="Resolve O'Neal asset variant references.",
)
async def oneal_storage_assets_refs(
    link_id: Optional[str] = None,
    collection_id: Optional[str] = None,
    object_id: Optional[int] = None,
    role: Optional[str] = None,
    mine: Optional[bool] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    params = _clean_params(
        link_id=link_id,
        collection_id=collection_id,
        object_id=object_id,
        role=role,
        mine=mine,
        limit=limit,
    )
    return await call_oneal_storage_api("GET", "/storage/asset-refs", params=params)


@oneal_storage_mcp.tool(
    name="kg_health",
    description="Fetch knowledge graph health diagnostics for O'Neal tenant.",
)
async def oneal_storage_kg_health() -> Dict[str, Any]:
    return await call_oneal_storage_api("GET", "/storage/kg/health")


@oneal_storage_mcp.tool(
    name="kg_search",
    description="""Semantic text search across O'Neal tenant storage objects.

    Natural language search powered by OpenAI embeddings. Understands synonyms, context, and visual concepts.
    Example: "red motocross gloves for mountain biking" finds relevant O'Neal products even without exact keyword matches.
    Searches within tenant_oneal_knowledge collection (615 O'Neal product embeddings).
    """,
)
async def oneal_storage_kg_search(
    query: str,
    limit: int = 10,
    collection_like: Optional[str] = None,
    mine: Optional[bool] = None,
) -> Dict[str, Any]:
    params = _clean_params(
        query=query,
        limit=limit,
        collection_like=collection_like,
        mine=mine,
    )
    return await call_oneal_storage_api("GET", "/storage/kg/search", params=params)


@oneal_storage_mcp.tool(
    name="assets_get_embedding_text",
    description="""Get the embedding text for an O'Neal storage object.

    The embedding text is a 400-1000 character description that combines all AI metadata into searchable text.
    It is converted to a 3072-dimensional vector for semantic search in the Knowledge Graph.

    Returns:
    - object_id: Storage object ID
    - title: Object title
    - embedding_text: Full embedding description
    - searchable_fields: Fields included in search index
    - char_count: Character count
    """,
)
async def oneal_storage_assets_get_embedding_text(id: int) -> Dict[str, Any]:  # noqa: A002
    return await call_oneal_storage_api("GET", f"/storage/objects/{id}/embedding-text")


@oneal_storage_mcp.tool(
    name="assets_update_embedding_text",
    description="""Update embedding text and regenerate Knowledge Graph vector for O'Neal object.

    Manually refine the searchable description. The system will:
    1. Save new text to ai_context_metadata
    2. Generate new 3072-dim vector (OpenAI text-embedding-3-large)
    3. Update Chroma Vector DB automatically in tenant_oneal_knowledge collection

    Use this to improve semantic search results by adding domain-specific keywords or refining descriptions.
    """,
)
async def oneal_storage_assets_update_embedding_text(
    id: int,  # noqa: A002
    embedding_text: str
) -> Dict[str, Any]:
    return await call_oneal_storage_api(
        "PUT",
        f"/storage/objects/{id}/embedding-text",
        json_body={"embedding_text": embedding_text}
    )


# Artrack MCP --------------------------------------------------------------
artrack_mcp = FastMCP(
    name="artrack-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@artrack_mcp.tool(
    name="tracks_list",
    description="List all tracks for the authenticated user.",
)
async def artrack_tracks_list() -> Dict[str, Any]:
    tracks = await call_artrack_api("GET", "/tracks/")
    return {"tracks": tracks, "total": len(tracks)}


@artrack_mcp.tool(
    name="track_pretty",
    description="""Get human-readable track overview.

    Returns a clean summary of all routes with:
    - Route name, description, color
    - Total length in km
    - POI count
    - Segment count

    No coordinates or technical metadata - just the essentials.
    """,
)
async def artrack_track_pretty(track_id: int) -> Dict[str, Any]:
    return await call_artrack_api("GET", f"/tracks/{track_id}/pretty")


@artrack_mcp.tool(
    name="routes_ids",
    description="Get list of route IDs for a track. Simple endpoint for iteration.",
)
async def artrack_routes_ids(track_id: int) -> Dict[str, Any]:
    return await call_artrack_api("GET", f"/tracks/{track_id}/routes-ids")


@artrack_mcp.tool(
    name="route_pretty",
    description="""Get human-readable route detail.

    Returns:
    - Route info with total length in km
    - POIs with name, description, type, and position (km)
    - Segments with name, description, start/end km, and length

    No coordinates, no technical metadata - just the essentials.
    """,
)
async def artrack_route_pretty(track_id: int, route_id: int) -> Dict[str, Any]:
    return await call_artrack_api("GET", f"/tracks/{track_id}/routes/{route_id}/pretty")


@artrack_mcp.tool(
    name="pois_pretty",
    description="""Get human-readable list of all POIs for a track.

    Returns all Points of Interest with:
    - id, name, description, type
    - Position along route (km)
    - Route assignment (route_id, route_name)

    Optional: Filter by route_id.
    """,
)
async def artrack_pois_pretty(
    track_id: int,
    route_id: Optional[int] = None,
) -> Dict[str, Any]:
    params = _clean_params(route_id=route_id)
    return await call_artrack_api("GET", f"/tracks/{track_id}/pois-pretty", params=params)


@artrack_mcp.tool(
    name="pois_near",
    description="""GPS-based POI lookup with route snapping. Optimized for AI audio guides.

    Given a GPS position (lat, lng), returns nearby POIs sorted by distance with rich metadata
    (knowledge texts, audio_id, illustration_id, category) and route-snapping info
    (snapped position + along_meters on the route).

    This is the recommended way to find POIs for context-aware audio guides — no need to
    download all POIs and filter client-side.

    Parameters:
    - track_id: Track to search (e.g. 30 for Tscheppaschlucht)
    - lat, lng: User's current GPS position
    - radius_m: Max distance in meters (default 200)
    - limit: Max POIs to return (default 10)
    - route_id: Optional - constrain to a specific route
    - direction: "ahead" | "behind" | None (None = all). Requires user to be on a route.
    - include_text: Include knowledge texts (default true; set false for smaller payload)

    Returns:
    {
      track_id, query, snap (snapped pos + along_meters), total_found,
      pois: [{id, name, lat, lng, distance_m, along_meters, ahead, knowledge,
              audio_id, illustration_id, ...}]
    }

    Example use cases:
    - Audio guide: "Bin ich nah bei einem POI? Gib mir die Texte zum Vorlesen."
    - Story trigger: "Welcher Story Point ist als nächstes auf meiner Route?"
    - Navigation hint: "Was ist 100m vor mir interessant?"
    """,
)
async def artrack_pois_near(
    track_id: int,
    lat: float,
    lng: float,
    radius_m: int = 200,
    limit: int = 10,
    route_id: Optional[int] = None,
    direction: Optional[str] = None,
    include_text: bool = True,
) -> Dict[str, Any]:
    params = _clean_params(
        lat=lat,
        lng=lng,
        radius_m=radius_m,
        limit=limit,
        route_id=route_id,
        direction=direction,
        include_text=include_text,
    )
    return await call_artrack_api("GET", f"/tracks/{track_id}/pois-near", params=params)


@artrack_mcp.tool(
    name="pois_near_pretty",
    description="""GPS-based POI lookup formatted as compact text for AI consumption.

    Same as pois_near but returns a token-efficient plain-text summary instead of JSON.
    Includes all 3 intelligence layers (Route, POI, Story) + Screen Points.

    Use this when you want to feed the result directly into an LLM prompt without
    parsing JSON. The format is structured Markdown that LLMs can easily reason about.

    Returns sections:
    - Route Intelligence (where on route, progress, next POI)
    - POI Intelligence (manual waypoints with knowledge texts)
    - Story Intelligence (story points + POIs with attached stories)
    - Screen Points / Media (videos, photos)
    """,
)
async def artrack_pois_near_pretty(
    track_id: int,
    lat: float,
    lng: float,
    radius_m: int = 200,
    limit: int = 5,
    route_id: Optional[int] = None,
) -> str:
    params = _clean_params(
        lat=lat, lng=lng, radius_m=radius_m, limit=limit, route_id=route_id,
    )
    # Plain-text endpoint — fetch as text directly
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(
            f"{ARTRACK_API_BASE}/tracks/{track_id}/pois-near-pretty",
            headers={"X-API-KEY": ARTRACK_API_KEY},
            params=params,
        )
        response.raise_for_status()
        return response.text


@artrack_mcp.tool(
    name="segments_pretty",
    description="""Get human-readable list of all segments for a track.

    Returns all segments with:
    - id, name, description
    - Start/end position (km)
    - Length (km)
    - Route assignment (route_id, route_name)
    - Complete flag (has both start and end markers)

    Optional: Filter by route_id.
    """,
)
async def artrack_segments_pretty(
    track_id: int,
    route_id: Optional[int] = None,
) -> Dict[str, Any]:
    params = _clean_params(route_id=route_id)
    return await call_artrack_api("GET", f"/tracks/{track_id}/segments-pretty", params=params)


@artrack_mcp.tool(
    name="routes_list",
    description="List all routes for a track with full metadata.",
)
async def artrack_routes_list(track_id: int) -> Dict[str, Any]:
    return await call_artrack_api("GET", f"/tracks/{track_id}/routes")


@artrack_mcp.tool(
    name="waypoint_get",
    description="""Get full waypoint/POI details including complete description text.

    Returns complete waypoint data:
    - id, track_id, waypoint_type
    - Full user_description (not truncated)
    - Coordinates (latitude, longitude, altitude)
    - metadata_json with title and other metadata
    - media attachments
    - processing and moderation status

    Use this to get the full narrative text for a POI.
    """,
)
async def artrack_waypoint_get(waypoint_id: int) -> Dict[str, Any]:
    return await call_artrack_api("GET", f"/waypoints/{waypoint_id}")


@artrack_mcp.tool(
    name="knowledge_get",
    description="""Get all audio guide narration texts for a track.

    Returns the complete knowledge object with:
    - config: persona, target_audience, language, tone, background_knowledge
    - routes: intro/outro narrative texts per route
    - segments: entry/exit narrative texts per segment
    - pois: approaching/at_poi narrative texts per POI
    - Audio cues with storage IDs and durations (if generated)

    Each narrative text includes: text, text_original, edited flag.
    Data is stored in content-api as doc_type 'artrack_narration'.
    """,
)
async def artrack_knowledge_get(track_id: int) -> Dict[str, Any]:
    return await call_artrack_api("GET", f"/tracks/{track_id}/knowledge")


@artrack_mcp.tool(
    name="waypoint_update",
    description="""Update a waypoint/POI.

    Parameters:
    - waypoint_id: ID of the waypoint to update (required)
    - title: Display name of the POI (stored in metadata_json)
    - description: Full description text (user_description field)
    - tags: List of tags (stored in metadata_json)
    - priority: Display importance from -1.0 to 1.0 (higher = more prominent)
    - metadata_json: Dict to merge into existing metadata. Supports deep merge for nested objects.
                     Use this for assets, category, subcategory, or any custom fields.

    Only provided fields are updated, others remain unchanged.

    Example: Set a POI icon via metadata_json
      waypoint_update(waypoint_id=24693, metadata_json={"assets": [{"id": 103948, "role": "icon"}]})

    Example: Set category
      waypoint_update(waypoint_id=24693, metadata_json={"category": "waterfall", "subcategory": "scenic"})
    """,
)
async def artrack_waypoint_update(
    waypoint_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    priority: Optional[float] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if title is not None:
        body["title"] = title
    if description is not None:
        body["description"] = description
    if tags is not None:
        body["tags"] = tags
    if priority is not None:
        body["priority"] = priority
    if metadata_json is not None:
        body["metadata_json"] = metadata_json
    return await call_artrack_api("PUT", f"/waypoints/{waypoint_id}", json_body=body)


@artrack_mcp.tool(
    name="waypoint_attach_storage",
    description="""Attach storage media to a waypoint/POI.

    Links one or more Storage API objects (images, audio, video) to a waypoint.
    Use storage.assets_upload or storage.assets_fetch first to get storage IDs.

    Parameters:
    - waypoint_id: ID of the waypoint (required)
    - storage_ids: List of Storage object IDs to attach (required)
    - media_type: Type hint — 'photo', 'audio', or 'video' (optional)
    """,
)
async def artrack_waypoint_attach_storage(
    waypoint_id: int,
    storage_ids: List[int],
    media_type: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"storageIds": storage_ids}
    if media_type:
        body["mediaType"] = media_type
    return await call_artrack_api("POST", f"/waypoints/{waypoint_id}/attach-storage", json_body=body)


@artrack_mcp.tool(
    name="route_update",
    description="""Update a route.

    Parameters:
    - track_id: Track ID (required)
    - route_id: Route ID (required)
    - name: Route display name
    - color: Route color (hex, e.g. '#FF5733')
    - description: Route description text
    - storage_object_ids: List of Storage IDs for route media (cover images etc.)

    Only provided fields are updated.
    """,
)
async def artrack_route_update(
    track_id: int,
    route_id: int,
    name: Optional[str] = None,
    color: Optional[str] = None,
    description: Optional[str] = None,
    storage_object_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if name is not None:
        body["name"] = name
    if color is not None:
        body["color"] = color
    if description is not None:
        body["description"] = description
    if storage_object_ids is not None:
        body["storage_object_ids"] = storage_object_ids
    return await call_artrack_api("PATCH", f"/tracks/{track_id}/routes/{route_id}", json_body=body)


@artrack_mcp.tool(
    name="track_update",
    description="""Update a track.

    Parameters:
    - track_id: Track ID (required)
    - name: Track name
    - description: Track description
    - visibility: 'public', 'followers', or 'private'
    - track_type: e.g. 'hiking', 'biking'
    - tags: List of tags
    - storage_object_ids: List of Storage IDs for track media

    Only provided fields are updated.
    """,
)
async def artrack_track_update(
    track_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    visibility: Optional[str] = None,
    track_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    storage_object_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    # Need current track data for required fields
    current = await call_artrack_api("GET", f"/tracks/{track_id}")
    body: Dict[str, Any] = {
        "name": name if name is not None else current.get("name", ""),
        "client_track_id": current.get("client_track_id", ""),
    }
    if description is not None:
        body["description"] = description
    if visibility is not None:
        body["visibility"] = visibility
    if track_type is not None:
        body["track_type"] = track_type
    if tags is not None:
        body["tags"] = tags
    if storage_object_ids is not None:
        body["storage_object_ids"] = storage_object_ids
    return await call_artrack_api("PUT", f"/tracks/{track_id}", json_body=body)


@artrack_mcp.tool(
    name="guide_config_update",
    description="""Update the audio guide configuration for a track.

    Parameters:
    - track_id: Track ID (required)
    - language: Language code (e.g. 'de-AT', 'en-US')
    - persona: Narrator persona/character description
    - user_type: Target audience (e.g. 'wanderer', 'family', 'sportler')
    - narrative_tone: Tone of narration (e.g. 'motivating', 'informative', 'poetic')
    - user_interests: List of user interests for personalization
    - default_radius: POI trigger radius in meters (default 15)

    Only provided fields are updated in the guide config.
    """,
)
async def artrack_guide_config_update(
    track_id: int,
    language: Optional[str] = None,
    persona: Optional[str] = None,
    user_type: Optional[str] = None,
    narrative_tone: Optional[str] = None,
    user_interests: Optional[List[str]] = None,
    default_radius: Optional[int] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if language is not None:
        body["language"] = language
    if persona is not None:
        body["persona"] = persona
    if user_type is not None:
        body["userType"] = user_type
    if narrative_tone is not None:
        body["narrativeTone"] = narrative_tone
    if user_interests is not None:
        body["userInterests"] = user_interests
    if default_radius is not None:
        body["defaultRadius"] = default_radius
    return await call_artrack_api("PUT", f"/guides/{track_id}/guide-config", json_body=body)


@artrack_mcp.tool(
    name="osm_nearby",
    description="""Query nearby real-world features from OpenStreetMap within a radius.

    Returns buildings, restaurants, shops, churches, parks, historic sites etc.
    that actually exist near the given GPS position. Use this to ground your
    narration in real surroundings — reference actual places instead of guessing.

    Results are server-side cached (1h, ~110m grid) so repeated calls are fast.

    Parameters:
    - lat, lng: GPS position to search around
    - radius_m: Search radius in meters (10-2000, default 200)

    Returns compact text: "Café Mozart (cafe, 15m) | Stadtpark (park, 45m) | ..."

    When to use:
    - User is walking/biking and you want to mention real nearby places
    - You need to verify if a building/shop/landmark actually exists nearby
    - User asks "what's around me?" and you want facts, not hallucinations
    """,
)
async def artrack_osm_nearby(
    lat: float,
    lng: float,
    radius_m: int = 200,
) -> Dict[str, Any]:
    return await call_artrack_api(
        "GET", "/osm/nearby/compact",
        params={"lat": lat, "lng": lng, "radius_m": min(radius_m, 2000)},
    )


@artrack_mcp.tool(
    name="osm_nearby_full",
    description="""Query nearby OSM features with full detail (JSON with coordinates).

    Same data as osm_nearby but returns structured JSON with lat/lng per feature.
    Use when you need coordinates for map display or distance calculations.
    """,
)
async def artrack_osm_nearby_full(
    lat: float,
    lng: float,
    radius_m: int = 200,
) -> Dict[str, Any]:
    return await call_artrack_api(
        "GET", "/osm/nearby",
        params={"lat": lat, "lng": lng, "radius_m": min(radius_m, 2000)},
    )


@artrack_mcp.tool(
    name="service_health",
    description="Health check for the Artrack API.",
)
async def artrack_service_health() -> Dict[str, Any]:
    return await call_artrack_api("GET", "/health")


# CodePilot MCP (Human-in-the-loop) ------------------------------------------
codepilot_mcp = FastMCP(
    name="codepilot-human",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@codepilot_mcp.tool(
    name="notify_human",
    description="""Send a notification to the human via Telegram.

    Use this to inform the human about:
    - Successful completion of a task
    - Errors or failures that occurred
    - Important status updates

    The message is sent immediately and does not wait for a response.
    """,
)
async def codepilot_notify_human(message: str, chat_id: Optional[str] = None) -> Dict[str, Any]:
    """Send notification to human via Telegram (routed through Comm API)."""
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "sent": False}

    try:
        body: Dict[str, Any] = {"message": message}
        if chat_id:
            body["chat_id"] = chat_id
        result = await call_comm_api(
            "POST",
            "/api/v1/telegram/interventions/notification",
            json_body=body,
        )
        return {"sent": result.get("sent", False), "message_id": result.get("message_id")}
    except Exception as e:
        logger.error("Failed to send notification: %s", e)
        return {"error": str(e), "sent": False}


@codepilot_mcp.tool(
    name="ask_human",
    description="""Ask the human a question via Telegram and wait for their response.

    Two modes:
    1. With options (buttons): ask_human("Deploy?", options=["Yes", "No"])
       → Shows inline buttons, returns the clicked option

    2. Without options (text input): ask_human("What color code should I use?")
       → User types free text response

    IMPORTANT: Only use this tool when explicitly requested in the prompt.
    Examples of explicit requests:
    - "frag mich wenn du soweit bist"
    - "ask me about the color"
    - "frag mich wenn du nicht weiter weißt"

    Returns the human's response or error if timeout/failure.
    Default timeout: 5 minutes.
    """,
)
async def codepilot_ask_human(
    question: str,
    options: Optional[List[str]] = None,
    timeout_seconds: int = 300,
    chat_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask human a question and wait for response (routed through Comm API)."""
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "response": None}

    try:
        # Determine endpoint based on options
        if options and len(options) > 0:
            # Approval request with buttons
            body: Dict[str, Any] = {
                "message": question,
                "options": options,
                "timeout_seconds": timeout_seconds,
            }
            if chat_id:
                body["chat_id"] = chat_id
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/approval",
                json_body=body,
            )
        else:
            # Text input request
            body = {
                "message": question,
                "timeout_seconds": timeout_seconds,
            }
            if chat_id:
                body["chat_id"] = chat_id
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/text-input",
                json_body=body,
            )

        request_id = create_result.get("id")
        if not request_id:
            return {"error": "Failed to create intervention request", "response": None}

        # Wait for response (poll with 60s intervals, API max is 120s)
        start_time = time.time()
        max_poll = 60

        while True:
            elapsed = time.time() - start_time
            remaining = timeout_seconds - elapsed

            if remaining <= 0:
                return {
                    "error": "Timeout waiting for human response",
                    "response": None,
                    "request_id": request_id,
                }

            poll_timeout = max(1, min(max_poll, int(remaining)))

            wait_result = await call_comm_api(
                "GET",
                f"/api/v1/telegram/interventions/{request_id}/wait",
                params={"timeout": poll_timeout},
                timeout=poll_timeout + 10,
            )

            status = wait_result.get("status")
            if status == "responded":
                return {
                    "response": wait_result.get("response") or wait_result.get("response_text"),
                    "responded_by": wait_result.get("responded_by"),
                    "request_id": request_id,
                }
            elif status in ("expired", "cancelled"):
                return {
                    "error": f"Request {status}",
                    "response": None,
                    "request_id": request_id,
                }
            # Still pending, continue polling

    except Exception as e:
        logger.error("Failed to ask human: %s", e)
        return {"error": str(e), "response": None}


@codepilot_mcp.tool(
    name="create_change_request",
    description="""Create a change request in CodePilot for the specified project.

    Required: description (str).
    Choose either:
      - project_id (int) OR
      - project_name (string; matches name or github_repo, case-insensitive)

    Optional: priority (low|medium|high), defaults to medium.
    """,
)
async def codepilot_create_change_request(
    description: str,
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    priority: str = "medium",
) -> Dict[str, Any]:
    """Create a CR via CodePilot API."""
    if not CODEPILOT_API_TOKEN:
        return {"created": False, "error": "CODEPILOT_API_TOKEN not configured"}

    if not project_id and not project_name:
        return {"created": False, "error": "project_id or project_name is required"}

    resolved_project_id = project_id
    if not resolved_project_id and project_name:
        match = await _find_project_by_name(project_name)
        if not match:
            return {"created": False, "error": f"project not found for name '{project_name}'"}
        resolved_project_id = match["id"]

    priority_value = priority.lower().strip() if priority else "medium"

    try:
        result = await call_codepilot_api(
            "POST",
            "/api/v1/change-requests/",
            json_body={
                "project_id": resolved_project_id,
                "description": description,
                "priority": priority_value,
            },
        )
        return {"created": True, "cr": result}
    except Exception as e:
        logger.error("Failed to create change request: %s", e)
        return {"created": False, "error": str(e)}


@codepilot_mcp.tool(
    name="list_projects",
    description="List accessible CodePilot projects (id, name, github_repo). Useful to pick a valid project_id.",
)
async def codepilot_list_projects() -> Dict[str, Any]:
    """Return projects visible to the service user."""
    if not CODEPILOT_API_TOKEN:
        return {"error": "CODEPILOT_API_TOKEN not configured", "projects": []}

    try:
        result = await call_codepilot_api("GET", "/api/v1/projects/")
        return {"projects": result.get("items", [])}
    except Exception as e:
        logger.error("Failed to list projects: %s", e)
        return {"error": str(e), "projects": []}


async def _find_project_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Find project by name or github_repo (case-insensitive, partial match allowed)."""
    if not CODEPILOT_API_TOKEN:
        return None

    try:
        result = await call_codepilot_api("GET", "/api/v1/projects/", params={"limit": 200})
        projects = result.get("items", [])
    except Exception as e:
        logger.error("Failed to fetch projects for name lookup: %s", e)
        return None

    needle = name.lower().strip()

    # Exact match first
    for proj in projects:
        if proj.get("name", "").lower() == needle or proj.get("github_repo", "").lower() == needle:
            return proj

    # Partial match
    for proj in projects:
        if needle in proj.get("name", "").lower() or needle in proj.get("github_repo", "").lower():
            return proj

    return None


@codepilot_mcp.tool(
    name="ask_supervisor",
    description="""Ask the supervisor agent a question and wait for their response.

    Use this when you (a worker agent) need guidance or a decision from the
    supervising agent. The supervisor will receive your question and can
    provide an answer.

    This enables hierarchical agent communication without human intervention.

    Args:
        session_id: Unique session ID (provided by supervisor when spawning you)
        question: Your question for the supervisor
        context: Optional additional context to help supervisor understand

    Returns:
        The supervisor's answer

    Example:
        ask_supervisor(
            session_id="abc123",
            question="Which design style should I use for this tech company?",
            context="Available: Aurora Dark, Crystalline Glass, Organic Flow"
        )
    """,
)
async def codepilot_ask_supervisor(
    session_id: str,
    question: str,
    context: Optional[str] = None,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Ask supervisor agent a question and wait for response."""
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as client:
            response = await client.post(
                f"{CODEPILOT_API_BASE}/api/v1/supervisor/ask",
                headers={"Authorization": f"Bearer {CODEPILOT_API_TOKEN}"} if CODEPILOT_API_TOKEN else {},
                json={
                    "session_id": session_id,
                    "question": question,
                    "context": context,
                },
                params={"timeout": timeout_seconds},
            )
            response.raise_for_status()
            result = response.json()

        return {
            "answer": result.get("answer"),
            "session_id": result.get("session_id"),
            "success": True,
        }

    except httpx.TimeoutException:
        logger.warning("Timeout waiting for supervisor answer")
        return {
            "error": "Timeout waiting for supervisor answer",
            "answer": None,
            "success": False,
        }
    except Exception as e:
        logger.error("Failed to ask supervisor: %s", e)
        return {
            "error": str(e),
            "answer": None,
            "success": False,
        }


# Content API MCP --------------------------------------------------------------
content_mcp = FastMCP(
    name="content-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@content_mcp.tool(
    name="posts_list",
    description="""List posts with optional filters.

    Returns paginated list of posts with:
    - id, title, slug, content_type, status
    - author_id, author_name
    - created_at, updated_at
    - media count, annotations count, blocks count

    Filters: status, author_id, content_type, doc_type, partner_id
    """,
)
async def content_posts_list(
    status: Optional[str] = None,
    author_id: Optional[str] = None,
    content_type: Optional[str] = None,
    doc_type: Optional[str] = None,
    partner_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    params = _clean_params(
        status=status,
        author_id=author_id,
        content_type=content_type,
        doc_type=doc_type,
        partner_id=partner_id,
        limit=limit,
        offset=offset,
    )
    return await call_content_api("GET", "/api/v1/posts/", params=params)


@content_mcp.tool(
    name="posts_get",
    description="""Get a single post by ID with full details.

    Returns complete post data including:
    - All post fields (id, title, slug, content, etc.)
    - media: Array of attached media items with URLs
    - annotations: Array of content annotations
    - blocks: Array of structured content blocks
    """,
)
async def content_posts_get(post_id: int) -> Dict[str, Any]:
    return await call_content_api("GET", f"/api/v1/posts/{post_id}")


@content_mcp.tool(
    name="posts_create",
    description="""Create a new post.

    Required: title (str)
    Optional: content, content_type (md|html|json), status (draft|published|archived),
              author_id, author_name, metadata_json
    """,
)
async def content_posts_create(
    title: str,
    content: Optional[str] = None,
    content_type: str = "md",
    status: str = "draft",
    author_id: Optional[str] = None,
    author_name: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = {
        "title": title,
        "content": content,
        "content_type": content_type,
        "status": status,
    }
    if author_id:
        body["author_id"] = author_id
    if author_name:
        body["author_name"] = author_name
    if metadata_json:
        body["metadata_json"] = metadata_json
    return await call_content_api("POST", "/api/v1/posts/", json_body=body)


@content_mcp.tool(
    name="posts_update",
    description="""Update an existing post.

    All fields are optional. Only provided fields will be updated.
    """,
)
async def content_posts_update(
    post_id: int,
    title: Optional[str] = None,
    content: Optional[str] = None,
    content_type: Optional[str] = None,
    status: Optional[str] = None,
    author_id: Optional[str] = None,
    author_name: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = _clean_params(
        title=title,
        content=content,
        content_type=content_type,
        status=status,
        author_id=author_id,
        author_name=author_name,
        metadata_json=metadata_json,
    )
    return await call_content_api("PUT", f"/api/v1/posts/{post_id}", json_body=body)


@content_mcp.tool(
    name="posts_content_patch",
    description="""Patch post content with line-level operations instead of full replacement.

    Much more efficient than posts_update for small changes — only sends the diff.

    Operations:
    - replace: Replace lines start_line..end_line with new text
    - insert: Insert text after start_line (use 0 for before first line)
    - delete: Remove lines start_line..end_line
    - find/replace: Find text within line range and replace it

    Example — make line 5 bold:
      posts_content_patch(post_id=587, operations=[
        {"op": "replace", "start_line": 5, "end_line": 5, "text": "**This line is now bold**"}
      ])

    Example — find and replace text in lines 10-20:
      posts_content_patch(post_id=587, operations=[
        {"op": "replace", "start_line": 10, "end_line": 20, "find": "old text", "replace_with": "new text"}
      ])

    Example — insert a new paragraph after line 8:
      posts_content_patch(post_id=587, operations=[
        {"op": "insert", "start_line": 8, "text": "\\nNew paragraph here.\\n"}
      ])

    Args:
        post_id: ID of the post to patch
        operations: List of patch operations (each has op, start_line, end_line, text, find, replace_with)
        author_id: Who made the change
        author_name: Display name of editor

    Returns:
        Updated post with new content
    """,
)
async def content_posts_content_patch(
    post_id: int,
    operations: List[Dict[str, Any]],
    author_id: Optional[str] = None,
    author_name: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "operations": operations,
    }
    if author_id:
        body["author_id"] = author_id
    if author_name:
        body["author_name"] = author_name
    return await call_content_api("PATCH", f"/api/v1/posts/{post_id}/content-patch", json_body=body)


@content_mcp.tool(
    name="posts_delete",
    description="Delete a post by ID. Returns success status.",
)
async def content_posts_delete(post_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/posts/{post_id}")


@content_mcp.tool(
    name="posts_export_pdf",
    description="""Export a post as a PDF document via WeasyPrint.

    Returns metadata + a publicly fetchable download_url. Set include_base64=True
    only when you really need the bytes inline (small posts only — large PDFs
    blow past the tool result token limit).

    Published posts: download_url is publicly accessible (shareable).
    Draft/archived: download_url requires the X-API-KEY header.

    Args:
        post_id: ID of the post to export
        include_media: Append the media gallery section (default True)
        include_base64: Inline the full PDF as base64 (default False — use the
                        download_url for sharing/large files)

    Returns:
        dict with:
        - post_id, filename, mime_type, size_bytes, is_published
        - download_url: PUBLIC URL to the PDF endpoint (auth-free for published posts)
        - pdf_base64: base64 PDF content (only if include_base64=True)
    """,
)
async def content_posts_export_pdf(
    post_id: int,
    include_media: bool = True,
    include_base64: bool = False,
) -> Dict[str, Any]:
    # Render via internal address (fast, no SSL handshake)
    internal_url = f"{CONTENT_API_BASE}/api/v1/posts/{post_id}/export.pdf"
    headers: Dict[str, str] = {}
    if CONTENT_API_KEY:
        headers["X-API-KEY"] = CONTENT_API_KEY
    params = {"include_media": "true" if include_media else "false"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(internal_url, headers=headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Upstream {response.status_code} from GET {internal_url}: {response.text[:200]}"
            )
        pdf_bytes = response.content
        disposition = response.headers.get("content-disposition", "")
        filename = f"post-{post_id}.pdf"
        if "filename=" in disposition:
            filename = disposition.split("filename=", 1)[1].strip().strip('"')

    # Public URL — works for sharing via Telegram / browser / etc.
    public_url = (
        f"{CONTENT_API_PUBLIC_BASE}/api/v1/posts/{post_id}/export.pdf"
        f"?include_media={'true' if include_media else 'false'}"
    )

    # Look up published status so caller knows if URL is auth-free
    try:
        post_meta = await call_content_api("GET", f"/api/v1/posts/{post_id}")
        is_published = post_meta.get("status") == "published"
    except Exception:
        is_published = False

    result: Dict[str, Any] = {
        "post_id": post_id,
        "filename": filename,
        "mime_type": "application/pdf",
        "size_bytes": len(pdf_bytes),
        "download_url": public_url,
        "is_published": is_published,
        "auth_required": not is_published,
    }
    if include_base64:
        result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
    return result


@content_mcp.tool(
    name="posts_export_themed_pdf",
    description="""Export a post as a themed PDF via the markdown-api.

    Available themes: default, minimal, technical, business, arkturian.
    The markdown-api renders the post with professional styling and typography.

    Returns metadata + a publicly fetchable download_url.

    Args:
        post_id: ID of the post to export
        theme: Visual theme (default, minimal, technical, business, arkturian)
        logo_url: Optional logo URL for branding (overrides post.logo_url).
                  Can be a Storage URL (e.g. https://api-storage.arkturian.com/storage/media/12345)
                  or any public image URL.
        include_base64: Inline the full PDF as base64 (default False)

    Returns:
        dict with:
        - post_id, theme, filename, mime_type, size_bytes, is_published
        - download_url: PUBLIC URL to the themed PDF endpoint
        - pdf_base64: base64 PDF content (only if include_base64=True)
    """,
)
async def content_posts_export_themed_pdf(
    post_id: int,
    theme: str = "default",
    logo_url: Optional[str] = None,
    include_base64: bool = False,
) -> Dict[str, Any]:
    internal_url = f"{CONTENT_API_BASE}/api/v1/posts/{post_id}/export-themed.pdf"
    headers: Dict[str, str] = {}
    if CONTENT_API_KEY:
        headers["X-API-KEY"] = CONTENT_API_KEY
    params: Dict[str, str] = {"theme": theme}
    if logo_url:
        params["logo_url"] = logo_url

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(internal_url, headers=headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Upstream {response.status_code}: {response.text[:200]}"
            )
        pdf_bytes = response.content
        disposition = response.headers.get("content-disposition", "")
        filename = f"post-{post_id}-{theme}.pdf"
        if "filename=" in disposition:
            filename = disposition.split("filename=", 1)[1].strip().strip('"')

    public_url = (
        f"{CONTENT_API_PUBLIC_BASE}/api/v1/posts/{post_id}/export-themed.pdf"
        f"?theme={theme}"
    )

    try:
        post_meta = await call_content_api("GET", f"/api/v1/posts/{post_id}")
        is_published = post_meta.get("status") == "published"
    except Exception:
        is_published = False

    result: Dict[str, Any] = {
        "post_id": post_id,
        "theme": theme,
        "filename": filename,
        "mime_type": "application/pdf",
        "size_bytes": len(pdf_bytes),
        "download_url": public_url,
        "is_published": is_published,
        "auth_required": not is_published,
    }
    if include_base64:
        result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
    return result


@content_mcp.tool(
    name="media_list",
    description="""List media attachments for a post.

    Returns all media items attached to the specified post with:
    - id, storage_id, position, caption, role
    - media_url: Direct URL to the media file
    - created_at
    """,
)
async def content_media_list(post_id: int) -> List[Dict[str, Any]]:
    return await call_content_api("GET", f"/api/v1/posts/{post_id}/media/")


@content_mcp.tool(
    name="media_add",
    description="""Add media attachment to a post.

    Required: storage_id (int) - ID from Storage API
    Optional: position (int), caption (str), role (str, e.g. 'hero', 'gallery')
    """,
)
async def content_media_add(
    post_id: int,
    storage_id: int,
    position: int = 0,
    caption: Optional[str] = None,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "storage_id": storage_id,
        "position": position,
    }
    if caption:
        body["caption"] = caption
    if role:
        body["role"] = role
    return await call_content_api("POST", f"/api/v1/posts/{post_id}/media/", json_body=body)


@content_mcp.tool(
    name="media_delete",
    description="Remove media attachment from a post.",
)
async def content_media_delete(post_id: int, media_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/posts/{post_id}/media/{media_id}")


@content_mcp.tool(
    name="annotations_list",
    description="""List annotations for a post.

    Annotations are metadata markers on content (highlights, comments, tags).
    Returns: id, annotation_type, target_selector, body_json, created_at
    """,
)
async def content_annotations_list(post_id: int) -> List[Dict[str, Any]]:
    return await call_content_api("GET", f"/api/v1/posts/{post_id}/annotations/")


@content_mcp.tool(
    name="annotations_create",
    description="""Create an annotation on a post.

    Required: annotation_type (str), target_selector (dict)
    Optional: body_json (dict) - annotation content/metadata
    """,
)
async def content_annotations_create(
    post_id: int,
    annotation_type: str,
    target_selector: Dict[str, Any],
    body_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = {
        "annotation_type": annotation_type,
        "target_selector": target_selector,
    }
    if body_json:
        body["body_json"] = body_json
    return await call_content_api("POST", f"/api/v1/posts/{post_id}/annotations/", json_body=body)


@content_mcp.tool(
    name="annotations_delete",
    description="Delete an annotation from a post.",
)
async def content_annotations_delete(post_id: int, annotation_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/posts/{post_id}/annotations/{annotation_id}")


@content_mcp.tool(
    name="blocks_list",
    description="""List content blocks for a post.

    Blocks are structured content units (text, image, video, embed, etc.).
    Returns: id, block_type, position, data_json, created_at
    """,
)
async def content_blocks_list(post_id: int) -> List[Dict[str, Any]]:
    return await call_content_api("GET", f"/api/v1/posts/{post_id}/blocks/")


@content_mcp.tool(
    name="blocks_create",
    description="""Create a content block in a post.

    Required: block_type (str, e.g. 'text', 'image', 'video', 'embed')
    Optional: position (int), data_json (dict) - block-specific data
    """,
)
async def content_blocks_create(
    post_id: int,
    block_type: str,
    position: int = 0,
    data_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = {
        "block_type": block_type,
        "position": position,
    }
    if data_json:
        body["data_json"] = data_json
    return await call_content_api("POST", f"/api/v1/posts/{post_id}/blocks/", json_body=body)


@content_mcp.tool(
    name="blocks_update",
    description="Update a content block. All fields optional.",
)
async def content_blocks_update(
    post_id: int,
    block_id: int,
    block_type: Optional[str] = None,
    position: Optional[int] = None,
    data_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = _clean_params(
        block_type=block_type,
        position=position,
        data_json=data_json,
    )
    return await call_content_api("PUT", f"/api/v1/posts/{post_id}/blocks/{block_id}", json_body=body)


@content_mcp.tool(
    name="blocks_delete",
    description="Delete a content block from a post.",
)
async def content_blocks_delete(post_id: int, block_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/posts/{post_id}/blocks/{block_id}")


@content_mcp.tool(
    name="references_list",
    description="""List references (bibliography entries) for a post.

    Returns all structured references with: id, ref_number, title, authors, url, publication, year, note, created_at
    References are ordered by ref_number.
    """,
)
async def content_references_list(post_id: int) -> List[Dict[str, Any]]:
    return await call_content_api("GET", f"/api/v1/posts/{post_id}/references/")


@content_mcp.tool(
    name="references_create",
    description="""Add a bibliographic reference to a post.

    Required: ref_number (int, e.g. 1), title (str)
    Optional: authors (str), url (str), publication (str), year (str), note (str)

    Use [N] in post content to create clickable citation links to ref_number N.
    """,
)
async def content_references_create(
    post_id: int,
    ref_number: int,
    title: str,
    authors: Optional[str] = None,
    url: Optional[str] = None,
    publication: Optional[str] = None,
    year: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"ref_number": ref_number, "title": title}
    if authors:
        body["authors"] = authors
    if url:
        body["url"] = url
    if publication:
        body["publication"] = publication
    if year:
        body["year"] = year
    if note:
        body["note"] = note
    return await call_content_api("POST", f"/api/v1/posts/{post_id}/references/", json_body=body)


@content_mcp.tool(
    name="references_update",
    description="Update a reference. All fields optional.",
)
async def content_references_update(
    post_id: int,
    reference_id: int,
    ref_number: Optional[int] = None,
    title: Optional[str] = None,
    authors: Optional[str] = None,
    url: Optional[str] = None,
    publication: Optional[str] = None,
    year: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    body = _clean_params(
        ref_number=ref_number,
        title=title,
        authors=authors,
        url=url,
        publication=publication,
        year=year,
        note=note,
    )
    return await call_content_api("PATCH", f"/api/v1/posts/{post_id}/references/{reference_id}", json_body=body)


@content_mcp.tool(
    name="references_delete",
    description="Delete a reference from a post.",
)
async def content_references_delete(post_id: int, reference_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/posts/{post_id}/references/{reference_id}")


@content_mcp.tool(
    name="stories_list",
    description="""List all stories in a given language.

    Stories are multilingual narrative content with scenes, narrators, and illustrations.
    Returns story cards (no full scene text) with available_languages info.

    Parameters:
    - language: Language code (de, en, sl, it, es). Default: de
    - status: Filter by status (draft, published, archived). Default: published
    """,
)
async def content_stories_list(
    language: str = "de",
    status: Optional[str] = "published",
) -> Dict[str, Any]:
    params = _clean_params(language=language, status=status)
    return await call_content_api("GET", "/api/v1/stories/", params=params)


@content_mcp.tool(
    name="stories_get",
    description="""Get a story by group ID and language.

    Returns the full story with all scenes including narrative text, mood, location,
    illustration prompts, and media references.

    If the requested language doesn't exist and auto_generate=True,
    the story is automatically translated from the source language (German).

    Parameters:
    - story_group: Story group identifier (shared across language variants)
    - language: Language code (de, en, sl, it, es). Default: de
    - auto_generate: Auto-translate if not available. Default: true
    """,
)
async def content_stories_get(
    story_group: str,
    language: str = "de",
    auto_generate: bool = True,
) -> Dict[str, Any]:
    params = _clean_params(language=language, auto_generate=auto_generate)
    return await call_content_api("GET", f"/api/v1/stories/{story_group}", params=params)


@content_mcp.tool(
    name="stories_languages",
    description="""List available languages for a story group.

    Returns:
    - languages: List of available language codes
    - supported_languages: All supported languages (de, en, sl, it, es)
    - missing_languages: Languages not yet translated
    """,
)
async def content_stories_languages(
    story_group: str,
) -> Dict[str, Any]:
    return await call_content_api("GET", f"/api/v1/stories/{story_group}/languages")


@content_mcp.tool(
    name="stories_create",
    description="""Create a new story with scenes.

    Creates a story with all metadata and scenes in one call. No need to know the
    internal JSON structure — just provide flat parameters.

    Parameters:
    - title: Story title (required)
    - story_group: Group ID shared across language variants (auto-generated from title if omitted)
    - language: Language code (de, en, sl, it, es). Default: de
    - subtitle: Subtitle text
    - narrator: Narrator name. Default: tschauko
    - color: Theme color hex. Default: #7CB342
    - cover: Cover image URL or storage ID
    - region: Region/location of the story
    - duration: Duration estimate (e.g. "12 min")
    - status: draft, published, or archived. Default: draft
    - author_id: Author identifier. Default: system
    - author_name: Author display name. Default: System
    - scenes: List of scene objects, each with:
        - title (str): Scene title
        - narrative (str): The narrative text
        - mood (str): Scene mood/atmosphere
        - location (str, optional): Scene location
        - illustrationPrompt (str, optional): AI prompt for illustration
    """,
)
async def content_stories_create(
    title: str,
    scenes: List[Dict[str, Any]],
    story_group: Optional[str] = None,
    language: str = "de",
    subtitle: str = "",
    narrator: str = "tschauko",
    color: str = "#7CB342",
    cover: str = "",
    region: str = "",
    duration: str = "",
    status: str = "draft",
    author_id: str = "system",
    author_name: str = "System",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "title": title,
        "language": language,
        "subtitle": subtitle,
        "narrator": narrator,
        "color": color,
        "cover": cover,
        "region": region,
        "duration": duration,
        "status": status,
        "author_id": author_id,
        "author_name": author_name,
        "scenes": scenes,
    }
    if story_group:
        body["story_group"] = story_group
    return await call_content_api("POST", "/api/v1/stories/", json_body=body)


@content_mcp.tool(
    name="stories_update",
    description="""Update an existing story.

    Only provided (non-null) fields are updated. If scenes are provided,
    they replace ALL existing scenes.

    Parameters:
    - story_group: Story group identifier (required)
    - language: Language code of the version to update. Default: de
    - title: New title (optional)
    - subtitle: New subtitle (optional)
    - narrator: New narrator name (optional)
    - color: New theme color hex (optional)
    - cover: New cover image (optional)
    - region: New region (optional)
    - duration: New duration (optional)
    - status: New status: draft, published, archived (optional)
    - scenes: New scenes list (replaces all existing scenes, optional)
    """,
)
async def content_stories_update(
    story_group: str,
    language: str = "de",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    narrator: Optional[str] = None,
    color: Optional[str] = None,
    cover: Optional[str] = None,
    region: Optional[str] = None,
    duration: Optional[str] = None,
    status: Optional[str] = None,
    scenes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body = _clean_params(
        title=title,
        subtitle=subtitle,
        narrator=narrator,
        color=color,
        cover=cover,
        region=region,
        duration=duration,
        status=status,
        scenes=scenes,
    )
    params = {"language": language}
    return await call_content_api("PUT", f"/api/v1/stories/{story_group}", json_body=body, params=params)


@content_mcp.tool(
    name="stories_generate_languages",
    description="""Batch-generate story translations for multiple languages.

    Translates the story from its source language (German) into the requested target languages.
    Only generates TEXT — no audio generation.

    Parameters:
    - story_group: Story group identifier
    - languages: List of target language codes (e.g. ["en", "sl", "it"])
    - force_refresh: Re-translate even if translation already exists. Default: false
    """,
)
async def content_stories_generate_languages(
    story_group: str,
    languages: List[str],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    return await call_content_api(
        "POST",
        f"/api/v1/stories/{story_group}/generate-languages",
        json_body={"languages": languages, "force_refresh": force_refresh},
    )


@content_mcp.tool(
    name="collections_list",
    description="""List collections (hierarchical tree support).

    Collections are hierarchical groupings for posts. Each collection has an
    optional parent (parent_id) — a collection without parent_id is top-level.
    Each collection knows post_count and children_count.

    Parameters:
    - parent_id: List only collections directly under this parent (int)
    - root_only: List only top-level collections (no parent). Default: false

    Common usage:
    - root_only=True               → top-level overview
    - parent_id=42                 → direct children of collection #42
    - (no params)                  → flat list of all collections (use for tree assembly)
    """,
)
async def content_collections_list(
    parent_id: Optional[int] = None,
    root_only: bool = False,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    if parent_id is not None:
        params["parent_id"] = parent_id
    if root_only:
        params["root_only"] = "true"
    return await call_content_api("GET", "/api/v1/collections/", params=params or None)


@content_mcp.tool(
    name="collections_tree",
    description="""Return the entire collection hierarchy as a nested tree.

    Convenience wrapper that calls collections_list with no filter and assembles
    a tree view (children nested under each parent). Each node carries:
    id, parent_id, name, slug, description, color, icon, position,
    post_count, children_count, children (list of child nodes).

    Use this when you want a single call to understand the full taxonomy.
    """,
)
async def content_collections_tree() -> List[Dict[str, Any]]:
    flat = await call_content_api("GET", "/api/v1/collections/")
    if not isinstance(flat, list):
        return []
    nodes: Dict[int, Dict[str, Any]] = {}
    for c in flat:
        nodes[c["id"]] = {**c, "children": []}
    roots: List[Dict[str, Any]] = []
    for c in flat:
        node = nodes[c["id"]]
        parent_id = c.get("parent_id")
        if parent_id and parent_id in nodes:
            nodes[parent_id]["children"].append(node)
        else:
            roots.append(node)
    return roots


@content_mcp.tool(
    name="collections_get",
    description="""Get a collection by slug with all its posts and direct children.

    Parameters:
    - slug: Collection slug (e.g. "adic26-defense")
    """,
)
async def content_collections_get(slug: str) -> Dict[str, Any]:
    return await call_content_api("GET", f"/api/v1/collections/{slug}")


@content_mcp.tool(
    name="collections_create",
    description="""Create a new collection — optionally as a child of another collection.

    Pass parent_id to nest the new collection under an existing one. Without
    parent_id the collection becomes top-level. Use this to build hierarchies
    like Engineering → Architecture → Decisions.

    Parameters:
    - name: Collection name (required)
    - parent_id: Parent collection ID — omit to create at top level
    - description: Optional description
    - color: Optional hex color (e.g. "#4CAF50")
    - icon: Optional icon name or emoji
    - position: Sort position within siblings (default 0)
    """,
)
async def content_collections_create(
    name: str,
    parent_id: Optional[int] = None,
    description: str = "",
    color: Optional[str] = None,
    icon: Optional[str] = None,
    position: Optional[int] = None,
) -> Dict[str, Any]:
    body = _clean_params(
        name=name,
        parent_id=parent_id,
        description=description,
        color=color,
        icon=icon,
        position=position,
    )
    return await call_content_api("POST", "/api/v1/collections/", json_body=body)


@content_mcp.tool(
    name="collections_update",
    description="""Update an existing collection.

    Use parent_id to MOVE a collection in the hierarchy (set to null to make
    it top-level, set to another id to nest it under that parent).

    Parameters:
    - collection_id: Collection ID (required)
    - name: New name (optional)
    - parent_id: New parent — pass an id to re-parent, or 0 / null to detach
    - description: New description (optional)
    - color: New color (optional)
    - icon: New icon (optional)
    - position: New sort position within siblings (optional)
    """,
)
async def content_collections_update(
    collection_id: int,
    name: Optional[str] = None,
    parent_id: Optional[int] = None,
    description: Optional[str] = None,
    color: Optional[str] = None,
    icon: Optional[str] = None,
    position: Optional[int] = None,
) -> Dict[str, Any]:
    body = _clean_params(
        name=name,
        description=description,
        color=color,
        icon=icon,
        position=position,
    )
    # parent_id needs special handling: explicit null means "detach to root",
    # so we keep the key even when value is None — but only if caller
    # explicitly passed it. Since FastMCP can't distinguish "not passed" from
    # "passed as None", treat 0 as the explicit-null sentinel.
    if parent_id is not None:
        body["parent_id"] = None if parent_id == 0 else parent_id
    return await call_content_api("PUT", f"/api/v1/collections/{collection_id}", json_body=body)


@content_mcp.tool(
    name="collections_delete",
    description="""Delete a collection. Does NOT delete the posts inside it.

    Parameters:
    - collection_id: Collection ID to delete
    """,
)
async def content_collections_delete(collection_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/collections/{collection_id}")


@content_mcp.tool(
    name="collections_add_post",
    description="""Add a post to a collection.

    Parameters:
    - collection_id: Collection ID
    - post_id: Post ID to add
    """,
)
async def content_collections_add_post(collection_id: int, post_id: int) -> Dict[str, Any]:
    return await call_content_api("POST", f"/api/v1/collections/{collection_id}/posts/{post_id}")


@content_mcp.tool(
    name="collections_remove_post",
    description="""Remove a post from a collection.

    Parameters:
    - collection_id: Collection ID
    - post_id: Post ID to remove
    """,
)
async def content_collections_remove_post(collection_id: int, post_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/collections/{collection_id}/posts/{post_id}")


@content_mcp.tool(
    name="service_health",
    description="Health check for the Content API.",
)
async def content_service_health() -> Dict[str, Any]:
    return await call_content_api("GET", "/health")


# --------------------------------------------------------------------------- #
# Tree API  –  Collaborative tree editing with node-level CRUD
# --------------------------------------------------------------------------- #

tree_mcp = FastMCP(
    name="tree-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@tree_mcp.resource(
    uri="docs://tree/guide",
    name="Tree API Guide",
    description="Read this first! Complete guide for the Tree API: data model, PM fields, workflows, and usage examples.",
)
async def tree_guide() -> str:
    return """# Tree API — AI Quick Reference

## Overview
Hierarchical tree editor for project planning, work breakdown structures (WBS),
and general-purpose nested data. Each project has one root node with unlimited
depth of children. Supports project management fields on every node.

## Data Model

### Project
- id, name, node_count, created_at, updated_at
- Each project has exactly one root node (auto-created)

### Node
| Field | Type | Description |
|-------|------|-------------|
| id | int | Auto-generated |
| parent_id | int | Parent node (null for root) |
| position | int | Sort order among siblings |
| name | string | **Required** — node title |
| description | string | Free text, supports markdown |
| start_date | string | ISO date (YYYY-MM-DD), e.g. "2026-04-01" |
| end_date | string | ISO date (YYYY-MM-DD), fixed end date (e.g. project deadline) |
| effort_pt | float | Effort in person-days (Personentage) |
| budget | float | Planned budget (decimal) |
| actual_cost | float | Actual cost (decimal) |
| budget_cap | float | Top-down budget constraint (max budget for subtree) |
| effort_cap | float | Top-down effort constraint (max effort PT for subtree) |
| capacity | float | Team capacity in FTE |
| weight | float | Relative weight for proportional allocation (default: 1) |
| timebox_start | string | Timebox start date (YYYY-MM-DD) |
| timebox_end | string | Timebox end date (YYYY-MM-DD) |
| node_type | string | null (=task), "milestone", "deliverable", or "phase" |
| status | string | null (=open), "open", "reached", "missed", "delayed" |
| metadata | dict | Arbitrary JSON key-value pairs |
| children | list | Nested child nodes (in tree responses) |

### Computed Fields (read-only, calculated by backend)
| Field | Type | Description |
|-------|------|-------------|
| computed_effort_pt | float | Sum of children's effort_pt (bottom-up) |
| computed_budget | float | Sum of children's budgets (bottom-up) |
| computed_cost | float | Sum of children's actual_cost (bottom-up) |
| computed_start | string | Earliest child start_date (bottom-up) |
| computed_end | string | Latest child end_date (bottom-up) |
| allocated_effort | float | Effort allocated from parent's effort_cap by weight |
| allocated_budget | float | Budget allocated from parent's effort_cap by weight |
| delta_effort | float | computed_effort_pt - effort_cap (positive = over) |
| delta_budget | float | computed_budget - budget_cap (positive = over) |

### PM Fields — How They Work
- **effort_pt** is the primary planning metric (person-days of work)
- **computed_effort_pt** is auto-aggregated from children on parent nodes
- **start_date** is set on AP/phase level for timeline planning
- Budget vs actual_cost: progress ratio = actual_cost / budget
- These fields are optional — nodes without them still work fine

### Top-Down Planning (Bidirectional)
The tree supports **bidirectional planning**: bottom-up aggregation AND top-down allocation.

**Bottom-up** (automatic): Parent nodes aggregate effort, budget, cost, dates from children.

**Top-down** (manual constraints): Set budget_cap/effort_cap on a parent to define limits.
The system distributes the cap proportionally to children based on their **weight** field.
- Formula: `allocated = parent_cap * (child.weight / sum_of_all_weights)`
- Default weight is 1 (equal distribution)
- **delta** shows over/undershoot: positive delta = constraint exceeded

**Example workflow:**
```
1. Set effort_cap=100 on a phase node with 3 children
2. Set weight=2 on child A, weight=1 on B and C (default)
3. Result: A gets allocated_effort=50, B and C get 25 each
4. If A has computed_effort_pt=60, delta_effort=+10 (over by 10 PT)
```

## Key Workflows

### 1. Create a Project with WBS
```
1. projects_create(name="My Project")        → get project with root_node_id
2. tree_outline(project_id)                  → see structure + node IDs
3. nodes_create(project_id, parent_id=root_id, name="Phase 1")
4. nodes_create(project_id, parent_id=phase1_id, name="Task 1.1",
     start_date="2026-04-01", effort_pt=5, budget=2000)
```

### 2. Assign PM Data to Existing Nodes
```
1. tree_outline(project_id)                  → browse the tree, find node IDs
2. nodes_update(node_id, start_date="2026-04-01", effort_pt=14.5,
     budget=5000, actual_cost=1200)
```

### 2b. Set Top-Down Constraints
```
1. tree_outline(project_id)                  → find the parent node
2. nodes_update(node_id, budget_cap=50000, effort_cap=100)
3. Optionally set weights on children:
   nodes_update(child_id, weight=2)          → gets 2x share
4. tree_compact(project_id)                  → verify allocated_* and delta_* fields
```

### 3. Import a Complete Tree (JSON)
```
tree_import(name="Project X", tree={
  "name": "Root",
  "children": [
    {"name": "Phase 1", "start_date": "2026-04-01", "effort_pt": 30, "children": [
      {"name": "Task 1.1", "effort_pt": 5, "budget": 1000},
      {"name": "Task 1.2", "effort_pt": 10, "budget": 2000}
    ]},
    {"name": "Phase 2", "start_date": "2026-05-01", "effort_pt": 20}
  ]
})
```

### 4. Reorganize Nodes
```
nodes_move(node_id, parent_id=new_parent, position=0)   → move + reorder
nodes_delete(node_id)                                     → cascade deletes children
```

### 5. Export for Backup
```
tree_export(project_id)   → clean JSON without internal IDs
```

## Milestones, Deliverables & Node Types

Nodes have an optional `node_type` field:
- **null** (default) = regular task/work package
- **"milestone"** = zero-duration checkpoint (e.g. "Kick-off", "Go-Live")
- **"deliverable"** = project result/output (e.g. "Report", "Prototype")
- **"phase"** = grouping container (visual distinction)

Milestones and deliverables also have a `status` field:
- **null / "open"** = not yet reached
- **"reached"** = milestone achieved / deliverable completed
- **"missed"** = deadline passed, not achieved
- **"delayed"** = behind schedule

**Milestone conventions:**
- Use `end_date` as the due date (not start_date)
- Don't set effort_pt or budget on milestones (they're zero-duration)
- Frontend shows ◆ diamond icon and colored status badge
- tree_outline shows `type=milestone` and `status=reached` tags

**Deliverable conventions:**
- No start_date or end_date needed (result, not a time-bounded activity)
- No effort_pt or budget
- Frontend shows ▣ icon
- FFG export: "Art des Elements" = "Deliverable", no dates

**Create a milestone:**
```
nodes_create(project_id, parent_id, name="Kick-off Meeting",
    node_type="milestone", end_date="2026-04-01")
```

**Create a deliverable:**
```
nodes_create(project_id, parent_id, name="Final Report",
    node_type="deliverable")
```

**Update status:**
```
nodes_update(node_id, status="reached")
```

## Tips
- Use tree_outline for quick overview, tree_compact for structure + numbers, tree_get only when you need descriptions/metadata
- nodes_get for full single node details (descriptions, metadata, dates)
- position=0 means first child, position=1 means second, etc.
- nodes_move can move across branches (reparent)
- Deleting a node deletes ALL children (cascade)
- Cannot delete the root node — delete the project instead
- effort_pt supports decimals: 0.5 = half day, 3.5 = 3.5 person-days
- budget/actual_cost have no currency — up to the user to define

## Persons & Assignments

### Person
| Field | Type | Description |
|-------|------|-------------|
| id | int | Auto-generated |
| project_id | int | Belongs to project |
| name | string | Person name |
| email | string | Optional email |
| hourly_rate | float | Hourly rate (€/h) |
| color | string | Color hex code (e.g. "#4CAF50") |

### Assignment
| Field | Type | Description |
|-------|------|-------------|
| id | int | Auto-generated |
| node_id | int | Assigned to this node |
| person_id | int | The person |
| person_name | string | Person name (joined) |
| hours | float | Optional planned hours |
| role | string | Optional role description |
| color | string | Person color (joined) |

### Person Workflows

**Create persons for a project:**
```
1. persons_create(project_id, name="Alex", hourly_rate=85, color="#4CAF50")
2. persons_create(project_id, name="Maria", hourly_rate=95, color="#2196F3")
3. persons_list(project_id)   → see all persons
```

**Assign persons to tasks:**
```
1. assign_person(node_id, person_id=1)              → assign Alex to task
2. assign_person(node_id, person_id=2, hours=40)    → assign Maria with 40h
3. tree_outline(project_id)                          → shows person= tag on nodes
```

**Remove assignment:**
```
unassign_person(assignment_id=1, project_id=3)
```

Assignments appear in `tree_get` and `tree_outline` as an `assignments` array on each node.
The frontend shows person badges next to nodes (colored pill). Double-click to select a person.

## Frontend
The tree is visualized at tree.arkturian.com with 10 views:
MindMap, TreeView, TidyTree, Sunburst, Radial, Icicle, Treemap, CirclePack, Force, Gantt.
The Gantt view shows timeline bars based on start_date + effort_pt.
Tooltips show constraint caps with delta indicators (red=over, green=ok).
The EditNodeDialog has a "Constraints" section for budget_cap, effort_cap, weight, capacity, timebox.
The "Person" toggle in toolbar shows person badges on nodes. "Persons" button opens the persons panel.
"""


# --- Projects ---

@tree_mcp.tool(
    name="projects_list",
    description="""List all tree projects.

    Returns a list of projects with:
    - id, name, node_count
    - created_at, updated_at
    """,
)
async def tree_projects_list() -> Any:
    return await call_tree_api("GET", "/api/v1/projects/")


@tree_mcp.tool(
    name="projects_get",
    description="Get a single project by ID.",
)
async def tree_projects_get(project_id: int) -> Any:
    return await call_tree_api("GET", f"/api/v1/projects/{project_id}")


@tree_mcp.tool(
    name="projects_create",
    description="""Create a new tree project with an empty root node.

    Required: name (str).
    Returns the project with its root_node_id.
    """,
)
async def tree_projects_create(name: str) -> Any:
    return await call_tree_api("POST", "/api/v1/projects/", json_body={"name": name})


@tree_mcp.tool(
    name="projects_delete",
    description="Delete a project and all its nodes.",
)
async def tree_projects_delete(project_id: int) -> Any:
    return await call_tree_api("DELETE", f"/api/v1/projects/{project_id}")


# --- Tree (full hierarchy) ---

@tree_mcp.tool(
    name="tree_get",
    description="""Get the full tree for a project as nested JSON.

    Returns recursive structure:
    {id, name, description, start_date, budget, actual_cost, effort_pt, computed_effort_pt, metadata, children: [...]}

    effort_pt: person-days stored on leaf nodes.
    computed_effort_pt: auto-aggregated sum of children's effort (on parent nodes).

    Use this for initial load or overview. For large trees, prefer node-level operations.
    """,
)
async def tree_get(project_id: int) -> Any:
    return await call_tree_api("GET", f"/api/v1/projects/{project_id}/tree")


@tree_mcp.tool(
    name="tree_compact",
    description="""Get a compact tree (no descriptions, no metadata).

    Returns nested JSON with only: id, name, PM fields (effort, budget, cost, dates),
    constraint fields (budget_cap, effort_cap, weight, capacity, timebox),
    and computed fields (computed_effort_pt, computed_budget, delta_effort, delta_budget, allocated_*).
    ~80% smaller than tree_get. Use this when you need structure + PM numbers but not full details.
    For full node details, use nodes_get(node_id) on specific nodes.
    """,
)
async def tree_compact(project_id: int) -> Any:
    tree = await call_tree_api("GET", f"/api/v1/projects/{project_id}/tree")

    def strip(node: dict) -> dict:
        compact = {"id": node["id"], "name": node["name"]}
        for key in ("start_date", "end_date", "effort_pt", "computed_effort_pt", "budget", "actual_cost",
                     "budget_cap", "effort_cap", "capacity", "weight", "timebox_start", "timebox_end",
                     "computed_budget", "computed_cost", "computed_start", "computed_end",
                     "allocated_effort", "allocated_budget", "delta_effort", "delta_budget"):
            if node.get(key) is not None:
                compact[key] = node[key]
        children = node.get("children", [])
        if children:
            compact["children"] = [strip(c) for c in children]
        return compact

    return strip(tree)


@tree_mcp.tool(
    name="tree_outline",
    description="""Get a plain-text outline of the tree structure.

    Returns indented text with node names, IDs, and PM data (if set).
    Extremely compact (~1-2K tokens even for large trees).
    Best for quick overview, finding node IDs, and understanding hierarchy.

    Example output:
      INSECTA Projekt (id=6)
        EDERA SAFETY (id=7)
          AP1 — Projektmanagement (id=8, budget=68000)
            T1.1 — Systemarchitektur (id=9)
    """,
)
async def tree_outline(project_id: int) -> Any:
    tree = await call_tree_api("GET", f"/api/v1/projects/{project_id}/tree")
    lines: list[str] = []

    def walk(node: dict, depth: int = 0) -> None:
        indent = "  " * depth
        tags: list[str] = [f"id={node['id']}"]
        # Node type and status
        ntype = node.get("node_type")
        if ntype:
            tags.append(f"type={ntype}")
        nstatus = node.get("status")
        if nstatus:
            tags.append(f"status={nstatus}")
        for key, label in [("start_date", "start"), ("end_date", "end"),
                           ("effort_pt", "effort"), ("computed_effort_pt", "Σeffort"),
                           ("budget", "budget"), ("actual_cost", "cost"),
                           ("budget_cap", "cap-budget"), ("effort_cap", "cap-effort"),
                           ("weight", "weight"), ("capacity", "capacity"),
                           ("delta_effort", "Δeffort"), ("delta_budget", "Δbudget"),
                           ("allocated_effort", "alloc-effort"), ("allocated_budget", "alloc-budget")]:
            val = node.get(key)
            if val is not None:
                tags.append(f"{label}={val}")
        # Show assigned persons
        assignments = node.get("assignments", [])
        if assignments:
            names = ", ".join(a["person_name"] for a in assignments)
            tags.append(f"person={names}")
        icon = "◆ " if ntype == "milestone" else "▣ " if ntype == "deliverable" else ""
        lines.append(f"{indent}{icon}{node['name']} ({', '.join(tags)})")
        for child in node.get("children", []):
            walk(child, depth + 1)

    walk(tree)
    return "\n".join(lines)


@tree_mcp.tool(
    name="tree_import",
    description="""Import a JSON tree into a new project.

    Required: name (str), tree (dict with 'name' and optional 'children', 'description', etc.)
    Creates the project and recursively inserts all nodes.
    """,
)
async def tree_import(name: str, tree: Dict[str, Any]) -> Any:
    return await call_tree_api("POST", "/api/v1/projects/import", json_body={"name": name, "tree": tree})


@tree_mcp.tool(
    name="tree_export",
    description="""Export a project tree as clean JSON (no internal IDs).

    Returns the tree structure suitable for backup or transfer.
    """,
)
async def tree_export(project_id: int) -> Any:
    return await call_tree_api("GET", f"/api/v1/projects/{project_id}/export")


# --- Nodes ---

@tree_mcp.tool(
    name="nodes_create",
    description="""Create a new node in a project.

    Required: project_id, parent_id, name.
    Optional: description, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD),
              budget (decimal), actual_cost (decimal), effort_pt (float, person-days),
              budget_cap (float), effort_cap (float), capacity (float, FTE),
              weight (float), timebox_start (YYYY-MM-DD), timebox_end (YYYY-MM-DD),
              node_type (null/milestone/phase), status (null/open/reached/missed/delayed),
              metadata (dict).

    Returns the created node with its ID.
    """,
)
async def tree_nodes_create(
    project_id: int,
    parent_id: int,
    name: str,
    description: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    budget: Optional[float] = None,
    actual_cost: Optional[float] = None,
    effort_pt: Optional[float] = None,
    cost_personnel: Optional[float] = None,
    cost_material: Optional[float] = None,
    cost_travel: Optional[float] = None,
    cost_subcontracting: Optional[float] = None,
    cost_equipment: Optional[float] = None,
    budget_cap: Optional[float] = None,
    effort_cap: Optional[float] = None,
    capacity: Optional[float] = None,
    weight: Optional[float] = None,
    timebox_start: Optional[str] = None,
    timebox_end: Optional[str] = None,
    node_type: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    body = {"parent_id": parent_id, "name": name}
    body.update(_clean_params(
        description=description,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
        actual_cost=actual_cost,
        effort_pt=effort_pt,
        cost_personnel=cost_personnel,
        cost_material=cost_material,
        cost_travel=cost_travel,
        cost_subcontracting=cost_subcontracting,
        cost_equipment=cost_equipment,
        budget_cap=budget_cap,
        effort_cap=effort_cap,
        capacity=capacity,
        weight=weight,
        timebox_start=timebox_start,
        timebox_end=timebox_end,
        node_type=node_type,
        status=status,
        metadata=metadata,
    ))
    return await call_tree_api("POST", f"/api/v1/projects/{project_id}/nodes", json_body=body)


@tree_mcp.tool(
    name="nodes_get",
    description="Get a single node by ID with all fields.",
)
async def tree_nodes_get(node_id: int) -> Any:
    return await call_tree_api("GET", f"/api/v1/nodes/{node_id}")


@tree_mcp.tool(
    name="nodes_update",
    description="""Update a node. Only provided fields are changed.

    Optional: name, description, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD),
              budget (decimal), actual_cost (decimal), effort_pt (float, person-days),
              budget_cap (float), effort_cap (float), capacity (float, FTE),
              weight (float), timebox_start (YYYY-MM-DD), timebox_end (YYYY-MM-DD),
              node_type (null/milestone/phase), status (null/open/reached/missed/delayed),
              metadata (dict).
    """,
)
async def tree_nodes_update(
    node_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    budget: Optional[float] = None,
    actual_cost: Optional[float] = None,
    effort_pt: Optional[float] = None,
    cost_personnel: Optional[float] = None,
    cost_material: Optional[float] = None,
    cost_travel: Optional[float] = None,
    cost_subcontracting: Optional[float] = None,
    cost_equipment: Optional[float] = None,
    budget_cap: Optional[float] = None,
    effort_cap: Optional[float] = None,
    capacity: Optional[float] = None,
    weight: Optional[float] = None,
    timebox_start: Optional[str] = None,
    timebox_end: Optional[str] = None,
    node_type: Optional[str] = None,
    status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    body = _clean_params(
        name=name,
        description=description,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
        actual_cost=actual_cost,
        effort_pt=effort_pt,
        cost_personnel=cost_personnel,
        cost_material=cost_material,
        cost_travel=cost_travel,
        cost_subcontracting=cost_subcontracting,
        cost_equipment=cost_equipment,
        budget_cap=budget_cap,
        effort_cap=effort_cap,
        capacity=capacity,
        weight=weight,
        timebox_start=timebox_start,
        timebox_end=timebox_end,
        node_type=node_type,
        status=status,
        metadata_json=metadata,
    )
    return await call_tree_api("PATCH", f"/api/v1/nodes/{node_id}", json_body=body)


@tree_mcp.tool(
    name="nodes_delete",
    description="""Delete a node and all its children (cascade).

    Returns {deleted_ids: [...]}.
    Cannot delete the root node.
    """,
)
async def tree_nodes_delete(node_id: int) -> Any:
    return await call_tree_api("DELETE", f"/api/v1/nodes/{node_id}")


@tree_mcp.tool(
    name="nodes_move",
    description="""Move a node to a new parent and/or position.

    Required: node_id, parent_id (new parent), position (0-based index among siblings).
    """,
)
async def tree_nodes_move(node_id: int, parent_id: int, position: int) -> Any:
    return await call_tree_api("PATCH", f"/api/v1/nodes/{node_id}/move", json_body={"parent_id": parent_id, "position": position})


# --- Persons ---

@tree_mcp.tool(
    name="persons_list",
    description="List all persons for a project. Returns id, name, email, hourly_rate, color.",
)
async def tree_persons_list(project_id: int) -> Any:
    return await call_tree_api("GET", f"/api/v1/projects/{project_id}/persons")


@tree_mcp.tool(
    name="persons_create",
    description="Create a person in a project. Required: name. Optional: email, hourly_rate, color.",
)
async def tree_persons_create(
    project_id: int,
    name: str,
    email: Optional[str] = None,
    hourly_rate: Optional[float] = None,
    color: Optional[str] = None,
) -> Any:
    body = {"name": name}
    if email is not None:
        body["email"] = email
    if hourly_rate is not None:
        body["hourly_rate"] = hourly_rate
    if color is not None:
        body["color"] = color
    return await call_tree_api("POST", f"/api/v1/projects/{project_id}/persons", json_body=body)


@tree_mcp.tool(
    name="persons_update",
    description="Update a person. Pass only fields to change: name, email, hourly_rate, color.",
)
async def tree_persons_update(
    person_id: int,
    name: Optional[str] = None,
    email: Optional[str] = None,
    hourly_rate: Optional[float] = None,
    color: Optional[str] = None,
) -> Any:
    body = _clean_params(name=name, email=email, hourly_rate=hourly_rate, color=color)
    return await call_tree_api("PATCH", f"/api/v1/persons/{person_id}", json_body=body)


@tree_mcp.tool(
    name="persons_delete",
    description="Delete a person (cascade deletes all their assignments).",
)
async def tree_persons_delete(person_id: int) -> Any:
    return await call_tree_api("DELETE", f"/api/v1/persons/{person_id}")


# --- Assignments ---

@tree_mcp.tool(
    name="assign_person",
    description="Assign a person to a node (task). Optional: hours, role.",
)
async def tree_assign_person(
    node_id: int,
    person_id: int,
    hours: Optional[float] = None,
    role: Optional[str] = None,
) -> Any:
    body: Dict[str, Any] = {"person_id": person_id}
    if hours is not None:
        body["hours"] = hours
    if role is not None:
        body["role"] = role
    return await call_tree_api("POST", f"/api/v1/nodes/{node_id}/assignments", json_body=body)


@tree_mcp.tool(
    name="unassign_person",
    description="Remove an assignment by its ID.",
)
async def tree_unassign_person(assignment_id: int, project_id: int = 0) -> Any:
    return await call_tree_api("DELETE", f"/api/v1/assignments/{assignment_id}?project_id={project_id}")


@tree_mcp.tool(
    name="service_health",
    description="Health check for the Tree API.",
)
async def tree_service_health() -> Any:
    return await call_tree_api("GET", "/health")


# --------------------------------------------------------------------------- #
# FastAPI wrapper
# --------------------------------------------------------------------------- #

app = FastAPI(title="arkturian-mcp", version="3.0.0", description="Arkturian MCP Aggregator with RBAC authentication")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Authentication Middleware — validates agent tokens from Auth-API
from auth import JWTAuthMiddleware
app.add_middleware(JWTAuthMiddleware)

# Mount MCP transports
storage_app = storage_mcp.streamable_http_app()
oneal_app = oneal_mcp.streamable_http_app()
oneal_storage_app = oneal_storage_mcp.streamable_http_app()
artrack_app = artrack_mcp.streamable_http_app()
codepilot_app = codepilot_mcp.streamable_http_app()
content_app = content_mcp.streamable_http_app()
tree_app = tree_mcp.streamable_http_app()
mount_mcp("storage", STORAGE_PATH, storage_app)
mount_mcp("oneal", ONEAL_PATH, oneal_app)
mount_mcp("oneal-storage", ONEAL_STORAGE_PATH, oneal_storage_app)
mount_mcp("artrack", ARTRACK_PATH, artrack_app)
mount_mcp("codepilot", CODEPILOT_PATH, codepilot_app)
mount_mcp("content", CONTENT_PATH, content_app)
mount_mcp("tree", TREE_PATH, tree_app)
ai_mcp = FastMCP(
    name="ai-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@ai_mcp.tool(
    name="claude_text",
    description="Call AI API /ai/claude for text or vision. Minimal params: prompt (string). Optional: system, max_tokens, temperature, model.",
)
async def ai_claude_text(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.7,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        body["model"] = model
    return await call_ai_api("POST", "/ai/claude", json_body=body)


@ai_mcp.tool(
    name="chatgpt_text",
    description="Call AI API /ai/chatgpt. Params: prompt, system?, max_tokens?, temperature?, model?",
)
async def ai_chatgpt_text(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.7,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        body["model"] = model
    return await call_ai_api("POST", "/ai/chatgpt", json_body=body)


@ai_mcp.tool(
    name="gemini_text",
    description="Call AI API /ai/gemini. Params: prompt, system?, max_tokens?, temperature?, model?",
)
async def ai_gemini_text(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.7,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        body["model"] = model
    return await call_ai_api("POST", "/ai/gemini", json_body=body)


@ai_mcp.tool(
    name="transcribe_audio",
    description="Proxy to AI API /ai/transcribe (expects audio file URL). Provide `file_url` pointing to accessible audio.",
)
async def ai_transcribe_audio(file_url: str) -> Dict[str, Any]:
    return await call_ai_api("POST", "/ai/transcribe", json_body={"file_url": file_url})


@ai_mcp.tool(
    name="genimage",
    description="""Generate an image from a text prompt using AI models.

    Returns a generated image stored in the Storage API with:
    - id: Storage object ID
    - image_url: URL to the generated image
    - storage_object_id: Same as id
    - model: Model name used
    - actual_model: Actual model ID

    Available models:
    - nano-banana (default): Fast image generation (gemini-2.5-flash-image)
    - nano-banana-pro: Best quality (gemini-3-pro-image-preview)
    - imagen-4: Google Imagen 4.0

    Example:
    genimage(prompt="a futuristic city at sunset", model="nano-banana")
    """,
)
async def ai_genimage(
    prompt: str,
    model: Optional[str] = "nano-banana",
    width: Optional[int] = 1024,
    height: Optional[int] = 1024,
    negative_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate image from text prompt."""
    body = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
    }
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    return await call_ai_api("POST", "/ai/genimage", json_body=body)


ai_app = ai_mcp.streamable_http_app()
mount_mcp("ai", AI_PATH, ai_app)

# --------------------------------------------------------------------------- #
# Knowledge MCP – AI-powered knowledge extraction from Storage objects
# --------------------------------------------------------------------------- #

knowledge_mcp = FastMCP(
    name="knowledge-api",
    streamable_http_path="/",
    stateless_http=True,
    log_level="INFO",
)


@knowledge_mcp.tool(
    name="knowledge_query",
    description="""Query AI-powered knowledge for a storage object.

    Generates or retrieves cached knowledge by asking a question about a storage object.
    Uses AI vision for images/videos. Results are cached by (storage_id, prompt_hash).

    Parameters:
    - storage_id: Source storage object ID
    - prompt: Question about the object (max 4000 chars)
    - model: AI model (claude, chatgpt, gemini). Default: claude
    - force_refresh: Force regeneration even if cached. Default: false
    """,
)
async def knowledge_query(
    storage_id: int,
    prompt: str,
    model: str = "claude",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    return await call_knowledge_api("POST", "/api/v1/knowledge/query", json_body={
        "storage_id": storage_id,
        "prompt": prompt,
        "model": model,
        "force_refresh": force_refresh,
    })


@knowledge_mcp.tool(
    name="knowledge_get",
    description="Get a specific knowledge item by its ID.",
)
async def knowledge_get(knowledge_id: int) -> Dict[str, Any]:
    return await call_knowledge_api("GET", f"/api/v1/knowledge/{knowledge_id}")


@knowledge_mcp.tool(
    name="knowledge_source",
    description="List all knowledge items generated for a specific source storage object.",
)
async def knowledge_source(storage_id: int, limit: int = 50) -> Dict[str, Any]:
    return await call_knowledge_api("GET", f"/api/v1/knowledge/source/{storage_id}", params={"limit": limit})


@knowledge_mcp.tool(
    name="knowledge_annotated_generate",
    description="""Generate annotated knowledge for an image/video.

    Analyzes the image with AI vision and creates:
    - Object title and description
    - Annotation points with coordinates (x,y normalized 0-1), labels, descriptions
    - 3D positions for GLB models

    Detail levels: simple (kids), standard (general), detailed (expert)

    Parameters:
    - storage_id: Source storage object ID (must be image/video)
    - detail_level: simple, standard, or detailed. Default: standard
    - max_annotations: Max annotation points (1-10). Default: 6
    - language: Language for descriptions. Default: de
    - force_refresh: Force regeneration. Default: false
    """,
)
async def knowledge_annotated_generate(
    storage_id: int,
    detail_level: str = "standard",
    max_annotations: int = 6,
    language: str = "de",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    return await call_knowledge_api("POST", "/api/v1/knowledge/annotated", json_body={
        "storage_id": storage_id,
        "detail_level": detail_level,
        "max_annotations": max_annotations,
        "language": language,
        "force_refresh": force_refresh,
    })


@knowledge_mcp.tool(
    name="knowledge_annotated_get",
    description="Get annotated knowledge by ID. Returns object title, description, and all annotations with coordinates.",
)
async def knowledge_annotated_get(knowledge_id: int) -> Dict[str, Any]:
    return await call_knowledge_api("GET", f"/api/v1/knowledge/annotated/{knowledge_id}")


@knowledge_mcp.tool(
    name="knowledge_annotated_update",
    description="""Update annotated knowledge content.

    Can update title, description, or annotation points. Audio references are preserved.

    Parameters:
    - knowledge_id: ID of the annotated knowledge to update
    - object_title: Updated title (optional)
    - object_description: Updated description (optional)
    - annotations: Updated annotation list (optional) - each with id, anchor {x,y}, label, description
    """,
)
async def knowledge_annotated_update(
    knowledge_id: int,
    object_title: Optional[str] = None,
    object_description: Optional[str] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if object_title is not None:
        body["object_title"] = object_title
    if object_description is not None:
        body["object_description"] = object_description
    if annotations is not None:
        body["annotations"] = annotations
    return await call_knowledge_api("PUT", f"/api/v1/knowledge/annotated/{knowledge_id}", json_body=body)


@knowledge_mcp.tool(
    name="knowledge_annotated_audio",
    description="""Generate audio for annotated knowledge.

    Modes:
    - all: Object + each annotation separately (best for AR apps)
    - object: Only the main description
    - combined: Single audio combining everything
    - <annotation_id>: Only a specific annotation

    Parameters:
    - knowledge_id: Annotated knowledge ID
    - mode: all, object, combined, or specific annotation ID. Default: all
    - voice: OpenAI voice (alloy, echo, fable, onyx, nova, shimmer). Default: nova
    - add_music: Add background music. Default: true
    """,
)
async def knowledge_annotated_audio(
    knowledge_id: int,
    mode: str = "all",
    voice: str = "nova",
    add_music: bool = True,
) -> Dict[str, Any]:
    return await call_knowledge_api("POST", f"/api/v1/knowledge/annotated/{knowledge_id}/audio", json_body={
        "mode": mode,
        "voice": voice,
        "add_music": add_music,
    })


@knowledge_mcp.tool(
    name="knowledge_audio_generate",
    description="""Generate audio (Hörbuch) from knowledge content.

    Converts knowledge text to audio with TTS, optional background music and SFX.

    Parameters:
    - knowledge_id: Knowledge storage ID to convert
    - add_music: Add background music. Default: true
    - language: Language code. Default: de
    - narrator_voice: OpenAI voice (alloy, echo, fable, onyx, nova, shimmer). Default: nova
    """,
)
async def knowledge_audio_generate(
    knowledge_id: int,
    add_music: bool = True,
    language: str = "de",
    narrator_voice: str = "nova",
) -> Dict[str, Any]:
    return await call_knowledge_api("POST", "/api/v1/knowledge/audio", json_body={
        "knowledge_id": knowledge_id,
        "add_music": add_music,
        "language": language,
        "narrator_voice": narrator_voice,
    })


@knowledge_mcp.tool(
    name="knowledge_service_health",
    description="Check Knowledge API health status.",
)
async def knowledge_service_health() -> Dict[str, Any]:
    return await call_knowledge_api("GET", "/health")


knowledge_app = knowledge_mcp.streamable_http_app()
mount_mcp("knowledge", KNOWLEDGE_PATH, knowledge_app)

# --------------------------------------------------------------------------- #
# Review MCP - AI-powered multi-perspective review orchestrator
# --------------------------------------------------------------------------- #

review_mcp = FastMCP(
    name="review-api",
    streamable_http_path="/",
    stateless_http=True,
    log_level="INFO",
)


def _clean_params(**kwargs: Any) -> Dict[str, Any]:
    """Remove None values from keyword arguments."""
    return {k: v for k, v in kwargs.items() if v is not None}


@review_mcp.resource(uri="docs://review/guide", name="Review API Guide")
async def review_guide() -> str:
    return """# Review Orchestrator - AI Quick Reference

## Quick Start
1. `reviews_quick_website(url="https://example.com")` - Full website review
2. `reviews_create(subject_type="document", subject_title="My Doc", perspectives=["structure","clarity"], auto_execute=true)` - Custom review
3. `reviews_get(review_id=1)` - Check results
4. `reviews_findings(review_id=1)` - Get detailed findings

## Subject Types
- website: Web pages and landing pages
- presentation: Pitch decks, slide decks
- document: PDFs, DOCX, Markdown documents
- app_screen: App UI screens and flows
- api_content: Content from Content API
- branding: Brand assets and visual identity

## Perspectives
- design: Visual design, aesthetics, color, typography
- structure: Information architecture, hierarchy, organization
- clarity: Readability, comprehension, conciseness
- storytelling: Narrative flow, audience engagement
- brand_impact: Brand consistency, memorability
- consistency: Cross-element consistency
- ux: User experience, navigation, accessibility
- content_quality: Writing quality, accuracy, tone
- developer_readiness: Technical spec, implementation readiness

## Goals
- improve: General quality improvement suggestions
- prioritize: Focus on highest-impact changes
- sharpen: Optimize for specific audience
- handoff: Prepare for developer/team handoff

## Tips
- Use `context` parameter to provide additional info (target audience, brand guidelines, etc.)
- Use `templates_list()` to see predefined review configurations
- Reviews run in the background - check status with `reviews_get()`
"""


@review_mcp.tool(
    name="reviews_create",
    description="Create a new review. Set auto_execute=true to start immediately. Perspectives: design, structure, clarity, storytelling, brand_impact, consistency, ux, content_quality, developer_readiness.",
)
async def review_reviews_create(
    subject_type: str,
    subject_title: str,
    perspectives: List[str],
    subject_url: Optional[str] = None,
    subject_data: Optional[Dict[str, Any]] = None,
    goal: str = "improve",
    context: Optional[str] = None,
    template_slug: Optional[str] = None,
    auto_execute: bool = False,
) -> Dict[str, Any]:
    return await call_review_api(
        "POST",
        "/api/v1/reviews",
        json_body={
            "subject_type": subject_type,
            "subject_title": subject_title,
            "perspectives": perspectives,
            "subject_url": subject_url,
            "subject_data": subject_data or {},
            "goal": goal,
            "context": context,
            "template_slug": template_slug,
            "auto_execute": auto_execute,
        },
    )


@review_mcp.tool(
    name="reviews_list",
    description="List reviews. Optional filters: status (pending/gathering/analyzing/synthesizing/completed/failed), subject_type, limit.",
)
async def review_reviews_list(
    status: Optional[str] = None,
    subject_type: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    return await call_review_api(
        "GET",
        "/api/v1/reviews",
        params=_clean_params(status=status, subject_type=subject_type, limit=limit),
    )


@review_mcp.tool(
    name="reviews_get",
    description="Get a review with summary, overall score, and priority actions.",
)
async def review_reviews_get(review_id: int) -> Dict[str, Any]:
    return await call_review_api("GET", f"/api/v1/reviews/{review_id}")


@review_mcp.tool(
    name="reviews_findings",
    description="Get all findings for a review. Optional filters: perspective, severity (critical/warning/info/positive).",
)
async def review_reviews_findings(
    review_id: int,
    perspective: Optional[str] = None,
    severity: Optional[str] = None,
) -> Dict[str, Any]:
    return await call_review_api(
        "GET",
        f"/api/v1/reviews/{review_id}/findings",
        params=_clean_params(perspective=perspective, severity=severity),
    )


@review_mcp.tool(
    name="reviews_execute",
    description="Trigger or re-trigger review execution. The review runs in the background.",
)
async def review_reviews_execute(review_id: int) -> Dict[str, Any]:
    return await call_review_api("POST", f"/api/v1/reviews/{review_id}/execute")


@review_mcp.tool(
    name="reviews_quick_website",
    description="One-shot website review. Provide a URL and optionally select perspectives and goal.",
)
async def review_reviews_quick_website(
    url: str,
    perspectives: Optional[List[str]] = None,
    goal: str = "improve",
    context: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"url": url, "goal": goal}
    if perspectives:
        body["perspectives"] = perspectives
    if context:
        body["context"] = context
    return await call_review_api("POST", "/api/v1/reviews/quick/website", json_body=body)


@review_mcp.tool(
    name="templates_list",
    description="List available review templates. Optional filter: subject_type.",
)
async def review_templates_list(
    subject_type: Optional[str] = None,
) -> Dict[str, Any]:
    return await call_review_api(
        "GET",
        "/api/v1/templates",
        params=_clean_params(subject_type=subject_type),
    )


@review_mcp.tool(
    name="templates_get",
    description="Get a review template by ID.",
)
async def review_templates_get(template_id: int) -> Dict[str, Any]:
    return await call_review_api("GET", f"/api/v1/templates/{template_id}")


review_app = review_mcp.streamable_http_app()
mount_mcp("review", REVIEW_PATH, review_app)

# --------------------------------------------------------------------------- #
# Tarot MCP - Tarot card reading tools
# --------------------------------------------------------------------------- #

tarot_mcp = FastMCP(
    name="tarot-api",
    streamable_http_path="/",
    stateless_http=True,
    log_level="INFO",
)


@tarot_mcp.tool(
    name="tarot_draw",
    description="""Draw Tarot cards from a full 78-card deck.

    The deck contains:
    - 22 Major Arcana cards (The Fool through The World)
    - 56 Minor Arcana cards (Wands, Cups, Swords, Pentacles)

    Returns drawn cards with their meanings, keywords, and positions.

    Parameters:
    - num_cards: Number of cards to draw (1-78). Ignored if spread_type is set.
    - spread_type: Optional spread type (single, three_card, celtic_cross, love, decision)
    - allow_reversed: Whether cards can appear reversed (default true)

    Example:
    tarot_draw(num_cards=3, spread_type="three_card")
    """,
)
async def tarot_draw(
    num_cards: int = 1,
    spread_type: Optional[str] = None,
    allow_reversed: bool = True,
) -> Dict[str, Any]:
    """Draw Tarot cards."""
    body = {
        "num_cards": num_cards,
        "allow_reversed": allow_reversed,
    }
    if spread_type:
        body["spread_type"] = spread_type
    return await call_tools_api("POST", "/api/v1/tarot/draw", json_body=body)


@tarot_mcp.tool(
    name="tarot_spreads",
    description="Get all available Tarot spread types with their descriptions and positions.",
)
async def tarot_spreads() -> Dict[str, Any]:
    """Get available spread types."""
    return await call_tools_api("GET", "/api/v1/tarot/spreads")


@tarot_mcp.tool(
    name="tarot_deck_info",
    description="Get information about the Tarot deck (total cards, suits, spread types).",
)
async def tarot_deck_info() -> Dict[str, Any]:
    """Get deck information."""
    return await call_tools_api("GET", "/api/v1/tarot/deck")


tarot_app = tarot_mcp.streamable_http_app()
mount_mcp("tarot", TAROT_PATH, tarot_app)

# --------------------------------------------------------------------------- #
# Business API MCP – Invoicing, Clients, Transactions, Documents
# --------------------------------------------------------------------------- #

business_mcp = FastMCP(
    name="business-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


# --- AI Documentation Resource ---


@business_mcp.resource(
    uri="docs://business/guide",
    name="Business API Guide",
    description="Read this first! Complete guide for the Business API: workflows, tools, data model, and usage examples.",
)
async def business_guide() -> str:
    return """# Business API — AI Quick Reference

## Overview
Multi-tenant business management for Alex Popovic (Arkturian).
Handles Honorarnoten (fee notes), Rechnungen (invoices), clients, transactions, and EUeR.
Austrian tax rules apply (Kleinunternehmerregelung, 0% VAT default).

## Key Workflows

### 1. Create & Send Honorarnote
```
1. clients_list(search="name")           → get client_id
2. create_honorarnote(client_id, hours, rate, description)  → creates doc + PDF
3. documents_send(doc_id)                → emails PDF to client, status → sent
4. documents_mark_paid(doc_id)           → status → paid, creates income transaction
```

### 2. Create & Send Invoice
```
1. clients_list(search="name")           → get client_id
2. create_invoice(client_id, items=[{description, quantity, unit_price}])
3. documents_send(doc_id)                → emails PDF to client
4. documents_mark_paid(doc_id)           → status → paid
```

### 3. Record Expense
```
transactions_create(tx_type="expense", amount=X, category="...", description="...")
```

### 4. Financial Overview
```
dashboard_summary()                      → YTD/MTD income, expenses, profit
dashboard_cashflow(year=2026)            → monthly breakdown
```

### 5. PayPal — Visibility on incoming payments
```
paypal_status()                          → token + webhook health
paypal_events_list(limit=20)             → last events received from PayPal
paypal_events_list(process_status="error") → events that failed to process
paypal_sync()                            → re-process pending events into transactions
paypal_transactions(days=30)             → recent settled transactions (Reporting API)
paypal_transactions(direction="out", limit=10) → last debits ("Abbuchungen")
paypal_transactions(transaction_id="X")  → one specific transaction
paypal_balance()                         → current PayPal balance per currency
```
PayPal captures + paid invoices auto-create income transactions
(category="paypal", external_provider="paypal", external_id=<capture_id>).
The (tenant_id, external_provider, external_id) tuple is unique → safe to retry.

NOTE: paypal_transactions/balance hit PayPal's Reporting API and only see
SETTLED transactions. Pending entries shown in the consumer Dashboard
Activity view are not visible until they clear (1-3 business days).

## Data Model

### Clients
- id, name, company, email, phone, address, zip, city, country, uid_number
- Each document is linked to a client

### Documents (Honorarnoten & Rechnungen)
- doc_type: "honorarnote" or "invoice"
- status: draft → sent → paid (or cancelled/overdue)
- doc_number: auto-generated (HN-YYYY-NNN or RE-YYYY-NNN)
- Has line items, totals, PDF, dates

### Transactions
- tx_type: "income" or "expense"
- Paid documents auto-create income transactions
- Manual expenses via transactions_create

### Categories
- Used for transaction categorization
- cat_type: "income" or "expense"

## Important Defaults
- VAT rate: 0% (Kleinunternehmerregelung — VAT exempt)
- Currency: EUR
- Due days: 14
- Country: AT (Austria)

## Document Lifecycle
```
draft  →  sent (via documents_send)  →  paid (via documents_mark_paid)
  ↓                                        ↓
cancelled                              creates income transaction
```

## Common Clients (frequently used)
Use clients_list() to get current list with IDs.

## Tips
- documents_send() uses the client's email by default. Override with recipient_email param.
- documents_regenerate_pdf() after template changes to update existing PDFs.
- dashboard_summary() gives a quick financial overview.
- All monetary amounts are in EUR.
- Dates use YYYY-MM-DD format.
"""


# --- Dashboard ---


@business_mcp.tool(
    name="dashboard_summary",
    description="Get dashboard summary: income/expense YTD and MTD, profit, open documents.",
)
async def business_dashboard_summary(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/dashboard/summary", api_key=api_key)


@business_mcp.tool(
    name="dashboard_cashflow",
    description="Get monthly cashflow data (income, expense, profit per month) for a given year.",
)
async def business_dashboard_cashflow(year: Optional[int] = None, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    params = _clean_params(year=year)
    return await call_business_api("GET", "/api/v1/dashboard/cashflow", params=params, api_key=api_key)


# --- Clients ---


@business_mcp.tool(
    name="clients_list",
    description="List clients. Optional search by name/company. Returns id, name, company, email, phone, address, city.",
)
async def business_clients_list(
    search: Optional[str] = None,
    limit: int = 50,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = _clean_params(search=search, limit=limit)
    return await call_business_api("GET", "/api/v1/clients/", params=params, api_key=api_key)


@business_mcp.tool(
    name="clients_get",
    description="Get a single client by ID with all details.",
)
async def business_clients_get(client_id: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", f"/api/v1/clients/{client_id}", api_key=api_key)


@business_mcp.tool(
    name="clients_create",
    description="Create a new client. Required: name. Optional: company, email, phone, address, zip, city, country, uid_number, notes.",
)
async def business_clients_create(
    name: str,
    company: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    zip: Optional[str] = None,
    city: Optional[str] = None,
    country: str = "AT",
    uid_number: Optional[str] = None,
    notes: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {"name": name, "country": country}
    for k, v in {"company": company, "email": email, "phone": phone, "address": address,
                  "zip": zip, "city": city, "uid_number": uid_number, "notes": notes}.items():
        if v is not None:
            body[k] = v
    return await call_business_api("POST", "/api/v1/clients/", json_body=body, api_key=api_key)


@business_mcp.tool(
    name="clients_update",
    description="Update an existing client. Pass only the fields you want to change. Includes CRM fields: lead_status, source, next_followup_at.",
)
async def business_clients_update(
    client_id: int,
    name: Optional[str] = None,
    company: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    zip: Optional[str] = None,
    city: Optional[str] = None,
    country: Optional[str] = None,
    uid_number: Optional[str] = None,
    notes: Optional[str] = None,
    lead_status: Optional[str] = None,
    source: Optional[str] = None,
    next_followup_at: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {}
    for k, v in {"name": name, "company": company, "email": email, "phone": phone,
                  "address": address, "zip": zip, "city": city, "country": country,
                  "uid_number": uid_number, "notes": notes, "lead_status": lead_status,
                  "source": source, "next_followup_at": next_followup_at}.items():
        if v is not None:
            body[k] = v
    return await call_business_api("PATCH", f"/api/v1/clients/{client_id}", json_body=body, api_key=api_key)


# --- CRM ---


@business_mcp.tool(
    name="clients_update_status",
    description="Update a client's lead status. Statuses: lead, prospect, active, inactive, lost.",
)
async def business_clients_update_status(
    client_id: int,
    lead_status: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return await call_business_api(
        "PATCH", f"/api/v1/clients/{client_id}",
        json_body={"lead_status": lead_status}, api_key=api_key,
    )


@business_mcp.tool(
    name="clients_log_interaction",
    description="Log a client interaction (call, email, meeting, note). Auto-updates last_contact_at on the client.",
)
async def business_clients_log_interaction(
    client_id: int,
    interaction_type: str,
    subject: str,
    description: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"interaction_type": interaction_type, "subject": subject}
    if description:
        body["description"] = description
    return await call_business_api(
        "POST", f"/api/v1/clients/{client_id}/interactions", json_body=body, api_key=api_key,
    )


@business_mcp.tool(
    name="clients_interactions",
    description="List recent interactions for a client. Returns type, subject, date.",
)
async def business_clients_interactions(
    client_id: int,
    limit: int = 20,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = _clean_params(limit=limit)
    return await call_business_api(
        "GET", f"/api/v1/clients/{client_id}/interactions", params=params, api_key=api_key,
    )


@business_mcp.tool(
    name="clients_pipeline",
    description="CRM pipeline overview: clients grouped by lead status (lead, prospect, active, inactive, lost) with counts.",
)
async def business_clients_pipeline(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/crm/pipeline", api_key=api_key)


@business_mcp.tool(
    name="clients_followups",
    description="List clients with overdue or upcoming followups (next 7 days). Returns overdue and upcoming lists.",
)
async def business_clients_followups(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/crm/followups", api_key=api_key)


@business_mcp.tool(
    name="clients_set_followup",
    description="Set next followup date for a client. Date format: YYYY-MM-DD.",
)
async def business_clients_set_followup(
    client_id: int,
    next_followup_at: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return await call_business_api(
        "PATCH", f"/api/v1/clients/{client_id}",
        json_body={"next_followup_at": next_followup_at}, api_key=api_key,
    )


# --- Documents ---


@business_mcp.tool(
    name="documents_list",
    description="List documents (Honorarnoten, Rechnungen). Filter by doc_type (honorarnote, invoice), status (draft, sent, paid, overdue, cancelled), client_id, year.",
)
async def business_documents_list(
    doc_type: Optional[str] = None,
    status: Optional[str] = None,
    client_id: Optional[int] = None,
    year: Optional[int] = None,
    limit: int = 50,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = _clean_params(doc_type=doc_type, status=status, client_id=client_id, year=year, limit=limit)
    return await call_business_api("GET", "/api/v1/documents/", params=params, api_key=api_key)


@business_mcp.tool(
    name="documents_get",
    description="Get a single document with all line items and client details.",
)
async def business_documents_get(doc_id: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", f"/api/v1/documents/{doc_id}", api_key=api_key)


@business_mcp.tool(
    name="create_honorarnote",
    description="Create a new Honorarnote. Required: client_id, hours, rate, description. Auto-generates PDF. Returns document with items.",
)
async def business_create_honorarnote(
    client_id: int,
    hours: float,
    rate: float,
    description: str,
    vat_rate: float = 0.0,
    issued_date: Optional[str] = None,
    due_days: int = 14,
    notes: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "client_id": client_id,
        "hours": hours,
        "rate": rate,
        "description": description,
        "vat_rate": vat_rate,
        "due_days": due_days,
    }
    if issued_date:
        body["issued_date"] = issued_date
    if notes:
        body["notes"] = notes
    return await call_business_api("POST", "/api/v1/documents/honorarnote", json_body=body, api_key=api_key)


@business_mcp.tool(
    name="create_invoice",
    description="Create a new invoice with line items. Required: client_id, items (list of {description, quantity, unit_price}). Auto-generates PDF.",
)
async def business_create_invoice(
    client_id: int,
    items: List[Dict[str, Any]],
    vat_rate: float = 0.0,
    issued_date: Optional[str] = None,
    due_days: int = 14,
    notes: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "client_id": client_id,
        "items": items,
        "vat_rate": vat_rate,
        "due_days": due_days,
    }
    if issued_date:
        body["issued_date"] = issued_date
    if notes:
        body["notes"] = notes
    return await call_business_api("POST", "/api/v1/documents/invoice", json_body=body, api_key=api_key)


@business_mcp.tool(
    name="documents_mark_paid",
    description="Mark a document as paid. Optional paid_date (YYYY-MM-DD, defaults to today). Automatically creates an income transaction.",
)
async def business_documents_mark_paid(
    doc_id: int,
    paid_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {}
    if paid_date:
        body["paid_date"] = paid_date
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/mark-paid", json_body=body, api_key=api_key)


@business_mcp.tool(
    name="documents_regenerate_pdf",
    description="Regenerate the PDF for an existing document (e.g. after template changes).",
)
async def business_documents_regenerate_pdf(doc_id: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/regenerate-pdf", api_key=api_key)


@business_mcp.tool(
    name="documents_send",
    description="Send a document (Honorarnote/Invoice) to client via email with PDF attachment. Optional: recipient_email override (otherwise uses client email). Updates status to 'sent'.",
)
async def business_documents_send(
    doc_id: int,
    recipient_email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {}
    if recipient_email:
        body["recipient_email"] = recipient_email
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/send", json_body=body, api_key=api_key)


@business_mcp.tool(
    name="documents_cancel",
    description="Cancel a document (status → cancelled).",
)
async def business_documents_cancel(doc_id: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/cancel", api_key=api_key)


# --- Transactions ---


@business_mcp.tool(
    name="transactions_list",
    description="List transactions. Filter by tx_type (income/expense), year, category. Returns date, amount, category, description.",
)
async def business_transactions_list(
    tx_type: Optional[str] = None,
    year: Optional[int] = None,
    category: Optional[str] = None,
    limit: int = 100,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = _clean_params(tx_type=tx_type, year=year, category=category, limit=limit)
    return await call_business_api("GET", "/api/v1/transactions", params=params, api_key=api_key)


@business_mcp.tool(
    name="transactions_create",
    description="Create a transaction. Required: tx_type (income/expense), amount. Optional: category, description, tx_date (YYYY-MM-DD), vat_rate, client_id, receipt_url (URL to receipt image).",
)
async def business_transactions_create(
    tx_type: str,
    amount: float,
    category: Optional[str] = None,
    description: Optional[str] = None,
    tx_date: Optional[str] = None,
    vat_rate: float = 0.0,
    client_id: Optional[int] = None,
    receipt_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body = {"tx_type": tx_type, "amount": amount, "vat_rate": vat_rate}
    for k, v in {"category": category, "description": description, "tx_date": tx_date, "client_id": client_id, "receipt_url": receipt_url}.items():
        if v is not None:
            body[k] = v
    return await call_business_api("POST", "/api/v1/transactions", json_body=body, api_key=api_key)


@business_mcp.tool(
    name="transactions_delete",
    description="Delete a transaction by ID. Only works for transactions not linked to documents.",
)
async def business_transactions_delete(tx_id: int, api_key: Optional[str] = None) -> Dict[str, str]:
    await call_business_api("DELETE", f"/api/v1/transactions/{tx_id}", api_key=api_key)
    return {"status": "deleted", "id": str(tx_id)}


# --- Categories ---


@business_mcp.tool(
    name="categories_list",
    description="List transaction categories. Optional filter by cat_type (income/expense).",
)
async def business_categories_list(cat_type: Optional[str] = None, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    params = _clean_params(cat_type=cat_type)
    return await call_business_api("GET", "/api/v1/categories", params=params, api_key=api_key)


# --- PayPal ---


@business_mcp.tool(
    name="paypal_status",
    description=(
        "PayPal integration health: OAuth token reachable? Mode (sandbox/live)? "
        "Webhook registered? Returns {mode, configured, token_ok, webhook_id, registered_webhooks}."
    ),
)
async def business_paypal_status(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/paypal/status", api_key=api_key)


@business_mcp.tool(
    name="paypal_events_list",
    description=(
        "List PayPal webhook events received by the business-api. Filter by "
        "process_status (received/ignored/processed/error) or event_type "
        "(e.g. PAYMENT.CAPTURE.COMPLETED). Use this to answer 'what came in?'."
    ),
)
async def business_paypal_events_list(
    process_status: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 50,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = _clean_params(
        process_status=process_status,
        event_type=event_type,
        limit=limit,
    )
    return await call_business_api(
        "GET", "/api/v1/paypal/events", params=params, api_key=api_key
    )


@business_mcp.tool(
    name="paypal_sync",
    description=(
        "Re-process pending PayPal events (those still in 'received' or 'error' "
        "state) and fold them into the transactions ledger. Returns counts of "
        "processed/skipped/errored events plus the IDs of created transactions."
    ),
)
async def business_paypal_sync(
    limit: int = 50,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    params = _clean_params(limit=limit)
    return await call_business_api(
        "POST", "/api/v1/paypal/sync", params=params, api_key=api_key
    )


@business_mcp.tool(
    name="paypal_transactions",
    description=(
        "List PayPal transactions from the merchant's account via the Reporting "
        "API. Use this to answer 'what came in / went out on PayPal?'. Default "
        "window is the last 30 days (max 31, PayPal API limit). "
        "direction='in' for credits, 'out' for debits, omit for both. "
        "transaction_status: S (success), P (pending), V (voided), D (denied). "
        "Pending Activity entries from the consumer Dashboard usually do not "
        "appear here until they settle. Each row: transaction_id, date, amount, "
        "currency, direction, fee, status, counterparty, subject."
    ),
)
async def business_paypal_transactions(
    days: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    direction: Optional[str] = None,
    transaction_status: Optional[str] = None,
    transaction_id: Optional[str] = None,
    limit: int = 100,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    params = _clean_params(
        days=days,
        start_date=start_date,
        end_date=end_date,
        direction=direction,
        transaction_status=transaction_status,
        transaction_id=transaction_id,
        limit=limit,
    )
    return await call_business_api(
        "GET", "/api/v1/paypal/transactions", params=params, api_key=api_key
    )


@business_mcp.tool(
    name="paypal_balance",
    description=(
        "Current PayPal balance(s) on the merchant account. One entry per "
        "currency held. Returns {account_number, as_of, balances:[{currency, "
        "available, total, primary}]}."
    ),
)
async def business_paypal_balance(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/paypal/balance", api_key=api_key)


# --- Service Health ---


@business_mcp.tool(
    name="service_health",
    description="Health check for Business API.",
)
async def business_service_health() -> Dict[str, Any]:
    return await call_business_api("GET", "/health")


business_app = business_mcp.streamable_http_app()
mount_mcp("business", BUSINESS_PATH, business_app)


# Comm API MCP -----------------------------------------------------------
comm_mcp = FastMCP(
    name="comm-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@comm_mcp.resource(
    uri="docs://comm/guide",
    name="Comm API Guide",
    description="Read this first! Guide for the Communication API: email, Telegram, interventions.",
)
async def comm_guide() -> str:
    return """# Comm API — AI Quick Reference

## Overview
Unified communication service for sending emails and Telegram messages.
Used by Business API for document delivery, and directly for notifications.

## Tools

### Sending Messages
- send_email(to, subject, body, source, template, template_data) — Send email
- send_telegram(message, chat_id, to) — Send Telegram message. Use 'to' for name-based sending (e.g. to="sabrina")
- send_message(channel, to, subject, body, source) — Unified send (email or telegram)
- notify_human(message) — Quick Telegram notification to admin

### Interactive
- ask_human(question, options, timeout_seconds) — Ask admin via Telegram, wait for response

### Contacts
- contacts_list(limit) — List all auto-registered Telegram contacts
- contacts_get(contact_id) — Get contact details
- incoming_messages(sender, chat_id, limit) — List incoming Telegram messages

### Google Calendar
- calendar_list_events(source, time_min, time_max, max_results) — List upcoming events
- calendar_create_event(summary, start, end, source, description, location, attendees, timezone) — Create event
- calendar_delete_event(event_id, source) — Delete event

### Gmail
- gmail_list_messages(source, query, max_results) — List emails
- gmail_get_message(source, message_id) — Get full email
- gmail_latest(source, query) — Most recent email
- gmail_send(source, to, subject, body) — Send email via Gmail
- gmail_mark_read(source, message_id) — Mark as read
- gmail_mark_unread(source, message_id) — Mark as unread

### Info
- list_sources() — Available email/telegram source identities
- message_history(channel, source, limit) — Sent message log
- service_health() — Health check

## Email Templates
Available templates (pass as 'template' param):
- honorarnote_send — Honorarnote delivery email
- invoice_send — Invoice delivery email
- payment_reminder — Payment reminder

## Sources
- "arkturian" — Default: alex@arkturian.com

## Tips
- notify_human() is fire-and-forget, good for status updates
- ask_human() blocks until response or timeout (default 5min)
- For document emails, use Business API documents_send() instead of send_email directly
"""


# --- Unified Send ---


@comm_mcp.tool(
    name="send_email",
    description="""Send an email through the Comm API.

    Source identities depend on the comm-api instance:

    - arkturian box: "arkturian" (SMTP, alex@arkturian.com),
      "apopovic" (Gmail API, apopovic.aut@gmail.com),
      "edera" (Gmail API, a.popovic@edera-safety.com), plus any other
      source configured via SOURCES + GMAIL_REFRESH_TOKEN_* env vars.

    - pdrei box: only "jascha" (Gmail API,
      jascha.popovic@bellevue-living.net) is wired. pdrei BLOCKS
      outbound SMTP (Linode), so sending with a non-Gmail source will
      TIMEOUT with a 502 from the comm-api.

    If you omit source, the instance's configured default is used
    (env COMM_DEFAULT_SOURCE, falls back to "arkturian"). On pdrei
    that default is "jascha" so send_email without an explicit source
    Just Works and routes via Gmail API.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Plain text body
        source: Optional source identity. Omit to use instance default.
        template: Optional template name (e.g. "honorarnote_send", "invoice_send", "payment_reminder", "notification")
        template_data: Optional dict of data for template rendering
    """,
)
async def comm_send_email(
    to: str,
    subject: str,
    body: str = "",
    source: Optional[str] = None,
    template: Optional[str] = None,
    template_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    json_body: Dict[str, Any] = {
        "to": to,
        "subject": subject,
        "body": body,
    }
    if source:
        json_body["source"] = source
    if template:
        json_body["template"] = template
    if template_data:
        json_body["template_data"] = template_data
    return await call_comm_api("POST", "/api/v1/email/send", json_body=json_body)


@comm_mcp.tool(
    name="send_telegram",
    description="""Send a Telegram message through the Comm API.

    Args:
        message: Message text (Markdown supported)
        chat_id: Optional Telegram chat ID (defaults to admin chat)
        to: Optional contact name to resolve to chat_id (e.g. "sabrina"). Uses fuzzy matching.
        bot: Optional bot identity to send through (e.g. "sabotnig" for the
            Martin Sabotnig channel via @AgentOSKittBot). When omitted, the
            recipient contact's preferred_bot is used; otherwise the
            deployment's default bot. Use this only when you need to override
            the natural routing — most outbound calls should leave it unset.
    """,
)
async def comm_send_telegram(
    message: str,
    chat_id: Optional[str] = None,
    to: Optional[str] = None,
    bot: Optional[str] = None,
) -> Dict[str, Any]:
    json_body: Dict[str, Any] = {"message": message}
    if chat_id:
        json_body["chat_id"] = chat_id
    if to:
        json_body["to"] = to
    if bot:
        json_body["bot"] = bot
    return await call_comm_api("POST", "/api/v1/telegram/send", json_body=json_body)


@comm_mcp.tool(
    name="send_telegram_document",
    description=(
        "Send a document (PDF, invoice, any file) via Telegram. "
        "Pass EXACTLY ONE of `url` or `data_base64`:\n"
        "\n"
        "  url: A REAL, PUBLICLY REACHABLE HTTPS URL that returns the "
        "document directly (no auth, no HTML page). Telegram's servers fetch "
        "it themselves. DO NOT make up a URL or construct one from an MCP "
        "base path — if you don't have an actual public URL on hand, upload "
        "the file to Storage API first (POST /storage/upload with is_public=true) "
        "and use the returned file_url, or use data_base64 instead.\n"
        "\n"
        "  data_base64: Raw bytes as standard base64. Use when you already "
        "have the bytes (e.g. from gmail_get_attachment, or you read a local "
        "file with `base64 -w0 file.pdf`).\n"
        "\n"
        "Addressing: `to` resolves a contact name fuzzily, or `chat_id` sends "
        "to a specific numeric id. Falls back to admin chat if neither is set. "
        "Optional `caption` adds text below the document. "
        "Telegram limit: 50 MB per document. "
        "On error you get a 400 with the exact Telegram reason (bad URL, too "
        "large, invalid chat) — read it, don't retry blindly."
    ),
)
async def comm_send_telegram_document(
    filename: str,
    to: Optional[str] = None,
    chat_id: Optional[str] = None,
    caption: Optional[str] = None,
    url: Optional[str] = None,
    data_base64: Optional[str] = None,
    bot: Optional[str] = None,
) -> Dict[str, Any]:
    if not url and not data_base64:
        return {"error": "Either url or data_base64 must be provided"}
    if url and data_base64:
        return {"error": "Provide only one of url or data_base64, not both"}

    json_body: Dict[str, Any] = {"filename": filename}
    if to:
        json_body["to"] = to
    if chat_id:
        json_body["chat_id"] = chat_id
    if caption:
        json_body["caption"] = caption
    if url:
        json_body["url"] = url
    if data_base64:
        json_body["data"] = data_base64
    if bot:
        json_body["bot"] = bot
    return await call_comm_api("POST", "/api/v1/telegram/send-document", json_body=json_body)


@comm_mcp.tool(
    name="send_message",
    description="""Send a message via any channel (unified endpoint).

    Args:
        channel: "email" or "telegram"
        to: Recipient (email address or Telegram chat_id)
        body: Message body
        source: Source identity (default: "arkturian")
        subject: Email subject (required for email, ignored for telegram)
        template: Optional template name
        template_data: Optional template rendering data
    """,
)
async def comm_send_message(
    channel: str,
    to: str,
    body: str,
    source: str = "arkturian",
    subject: Optional[str] = None,
    template: Optional[str] = None,
    template_data: Optional[Dict[str, Any]] = None,
    bot: Optional[str] = None,
) -> Dict[str, Any]:
    json_body: Dict[str, Any] = {
        "channel": channel,
        "source": source,
        "to": to,
        "body": body,
    }
    if subject:
        json_body["subject"] = subject
    if template:
        json_body["template"] = template
    if template_data:
        json_body["template_data"] = template_data
    if bot:
        json_body["bot"] = bot
    return await call_comm_api("POST", "/api/v1/send", json_body=json_body)


# --- Interventions (Human-in-the-loop) ---


@comm_mcp.tool(
    name="notify_human",
    description="""Send a notification to the human via Telegram.

    Use this to inform the human about:
    - Successful completion of a task
    - Errors or failures that occurred
    - Important status updates

    The message is sent immediately and does not wait for a response.
    """,
)
async def comm_notify_human(
    message: str,
    bot: Optional[str] = None,
) -> Dict[str, Any]:
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "sent": False}
    try:
        body: Dict[str, Any] = {"message": message}
        if bot:
            body["bot"] = bot
        result = await call_comm_api(
            "POST",
            "/api/v1/telegram/interventions/notification",
            json_body=body,
        )
        return {"sent": result.get("sent", False), "message_id": result.get("message_id")}
    except Exception as e:
        logger.error("Failed to send notification: %s", e)
        return {"error": str(e), "sent": False}


@comm_mcp.tool(
    name="ask_human",
    description="""Ask the human a question via Telegram and wait for their response.

    Two modes:
    1. With options (buttons): ask_human("Deploy?", options=["Yes", "No"])
    2. Without options (text input): ask_human("What should I do?")

    Returns the human's response or error if timeout.
    Default timeout: 5 minutes.
    """,
)
async def comm_ask_human(
    question: str,
    options: Optional[List[str]] = None,
    timeout_seconds: int = 300,
    bot: Optional[str] = None,
) -> Dict[str, Any]:
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "response": None}
    try:
        body: Dict[str, Any]
        if options and len(options) > 0:
            body = {
                "message": question,
                "options": options,
                "timeout_seconds": timeout_seconds,
            }
            if bot:
                body["bot"] = bot
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/approval",
                json_body=body,
            )
        else:
            body = {
                "message": question,
                "timeout_seconds": timeout_seconds,
            }
            if bot:
                body["bot"] = bot
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/text-input",
                json_body=body,
            )

        request_id = create_result.get("id")
        if not request_id:
            return {"error": "Failed to create intervention request", "response": None}

        start_time = time.time()
        max_poll = 60

        while True:
            elapsed = time.time() - start_time
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                return {"error": "Timeout waiting for human response", "response": None, "request_id": request_id}

            poll_timeout = max(1, min(max_poll, int(remaining)))
            wait_result = await call_comm_api(
                "GET",
                f"/api/v1/telegram/interventions/{request_id}/wait",
                params={"timeout": poll_timeout},
                timeout=poll_timeout + 10,
            )

            status = wait_result.get("status")
            if status == "responded":
                return {
                    "response": wait_result.get("response") or wait_result.get("response_text"),
                    "responded_by": wait_result.get("responded_by"),
                    "request_id": request_id,
                }
            elif status in ("expired", "cancelled"):
                return {"error": f"Request {status}", "response": None, "request_id": request_id}

    except Exception as e:
        logger.error("Failed to ask human: %s", e)
        return {"error": str(e), "response": None}


# --- Contacts ---


@comm_mcp.tool(
    name="contacts_list",
    description="""List all Telegram contacts. Contacts are auto-registered when someone messages the bot.

    Returns: list of contacts with name, username, chat_id, last_seen.
    """,
)
async def comm_contacts_list(
    limit: int = 100,
) -> Dict[str, Any]:
    return await call_comm_api("GET", "/api/v1/contacts", params={"limit": limit})


@comm_mcp.tool(
    name="contacts_get",
    description="""Get a single contact by ID.

    Returns: contact details including telegram_id, chat_id, display_name, username, notes.
    """,
)
async def comm_contacts_get(
    contact_id: int,
) -> Dict[str, Any]:
    return await call_comm_api("GET", f"/api/v1/contacts/{contact_id}")


@comm_mcp.tool(
    name="incoming_messages",
    description="""List incoming Telegram messages received by the bot.

    Each message includes a `message_type` ("text", "photo", "document",
    "video", "voice", "audio", "sticker"). For media messages the response
    also carries `file_id`, `mime_type`, `file_size`, and `file_name`.
    Use `telegram_get_file(file_id)` to fetch the actual bytes when you
    need to process a photo/document/video.

    Optional filters:
    - sender: Filter by sender name (fuzzy match)
    - chat_id: Filter by chat ID
    - limit: Max results (default 50)
    """,
)
async def comm_incoming_messages(
    sender: Optional[str] = None,
    chat_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    params = _clean_params(sender=sender, chat_id=chat_id, limit=limit)
    return await call_comm_api("GET", "/api/v1/contacts/incoming", params=params)


@comm_mcp.tool(
    name="telegram_get_file",
    description="""Resolve a Telegram file_id to its actual content.

    Pass a `file_id` from `incoming_messages` (for photo/document/video/
    voice/audio/sticker rows) and get back the file metadata plus the raw
    bytes as base64. Use this to actually load the photo/document/video
    that arrived via Telegram into your processing pipeline.

    Returns: {file_id, file_unique_id, file_path, file_size, mime_type, data}
    where `data` is the base64-encoded raw bytes.
    """,
)
async def comm_telegram_get_file(file_id: str) -> Dict[str, Any]:
    return await call_comm_api("GET", f"/api/v1/telegram/files/{file_id}")


# --- Info ---


@comm_mcp.tool(
    name="list_sources",
    description="List available communication source identities.",
)
async def comm_list_sources() -> List[Dict[str, Any]]:
    return await call_comm_api("GET", "/api/v1/sources")


@comm_mcp.tool(
    name="message_history",
    description="Get sent message history. Optional filters: channel (email/telegram), source, limit.",
)
async def comm_message_history(
    channel: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    params = _clean_params(channel=channel, source=source, limit=limit)
    return await call_comm_api("GET", "/api/v1/messages", params=params)


@comm_mcp.tool(
    name="service_health",
    description="Health check for Comm API.",
)
async def comm_service_health() -> Dict[str, Any]:
    return await call_comm_api("GET", "/health")


@comm_mcp.tool(
    name="gmail_list_accounts",
    description="List configured Gmail accounts with their email addresses.",
)
async def comm_gmail_list_accounts() -> Dict[str, Any]:
    return await call_comm_api("GET", "/api/v1/gmail/accounts")


@comm_mcp.tool(
    name="gmail_list_messages",
    description=(
        "List emails for a Gmail account. "
        "Source: 'apopovic' (apopovic.aut@gmail.com) or 'edera' (a.popovic@edera-safety.com). "
        "Query examples: 'is:unread', 'from:someone@email.com', 'subject:Rechnung', 'newer_than:1d'."
    ),
)
async def comm_gmail_list_messages(
    source: str,
    query: str = "",
    max_results: int = 10,
) -> Dict[str, Any]:
    params = f"?query={query}&max_results={max_results}" if query else f"?max_results={max_results}"
    return await call_comm_api("GET", f"/api/v1/gmail/{source}/messages{params}")


@comm_mcp.tool(
    name="gmail_get_message",
    description=(
        "Get a single email with full body text and attachment metadata. "
        "Returns body plus an `attachments` list with "
        "{filename, mime_type, attachment_id, size}. "
        "Use gmail_get_attachment to download a specific attachment's bytes."
    ),
)
async def comm_gmail_get_message(source: str, message_id: str) -> Dict[str, Any]:
    return await call_comm_api("GET", f"/api/v1/gmail/{source}/messages/{message_id}")


@comm_mcp.tool(
    name="gmail_get_attachment",
    description=(
        "Download an email attachment as base64. "
        "Use gmail_get_message first to discover attachment_ids. "
        "Returns {account, message_id, attachment_id, size, data} where "
        "`data` is standard base64 (decode with base64.b64decode to get bytes)."
    ),
)
async def comm_gmail_get_attachment(
    source: str,
    message_id: str,
    attachment_id: str,
) -> Dict[str, Any]:
    return await call_comm_api(
        "GET",
        f"/api/v1/gmail/{source}/messages/{message_id}/attachments/{attachment_id}",
    )


@comm_mcp.tool(
    name="gmail_latest",
    description=(
        "Get the most recent email for a Gmail account. "
        "Source: 'apopovic' or 'edera'. Optional query filter."
    ),
)
async def comm_gmail_latest(source: str, query: str = "") -> Dict[str, Any]:
    params = f"?query={query}" if query else ""
    return await call_comm_api("GET", f"/api/v1/gmail/{source}/latest{params}")


@comm_mcp.tool(
    name="gmail_send",
    description=(
        "Send an email via Gmail. "
        "Source identifies which Gmail account sends. Available sources depend on the deployment "
        "(e.g. 'apopovic' = apopovic.aut@gmail.com, 'edera' = a.popovic@edera-safety.com, "
        "'jascha' = jascha.popovic@bellevue-living.net on pdrei)."
    ),
)
async def comm_gmail_send(
    source: str,
    to: str,
    subject: str,
    body: str,
    html: bool = False,
) -> Dict[str, Any]:
    return await call_comm_api("POST", f"/api/v1/gmail/{source}/send", json_body={
        "to": to, "subject": subject, "body": body, "html": html
    })


@comm_mcp.tool(
    name="gmail_mark_read",
    description="Mark a Gmail message as read.",
)
async def comm_gmail_mark_read(source: str, message_id: str) -> Dict[str, Any]:
    return await call_comm_api("POST", f"/api/v1/gmail/{source}/mark-read/{message_id}")


@comm_mcp.tool(
    name="gmail_mark_unread",
    description="Mark a Gmail message as unread.",
)
async def comm_gmail_mark_unread(source: str, message_id: str) -> Dict[str, Any]:
    return await call_comm_api("POST", f"/api/v1/gmail/{source}/mark-unread/{message_id}")


@comm_mcp.tool(
    name="calendar_list_events",
    description=(
        "List upcoming Google Calendar events. "
        "Source: 'apopovic' or 'edera'. "
        "Optional: time_min/time_max (ISO format), max_results."
    ),
)
async def comm_calendar_list_events(
    source: str = "apopovic",
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    params = _clean_params(time_min=time_min, time_max=time_max, max_results=max_results)
    return await call_comm_api("GET", f"/api/v1/calendar/{source}/events", params=params)


@comm_mcp.tool(
    name="calendar_create_event",
    description=(
        "Create a Google Calendar event. "
        "Source: 'apopovic' or 'edera'. "
        "start/end: ISO datetime 'YYYY-MM-DDTHH:MM:SS' or date 'YYYY-MM-DD' for all-day. "
        "attendees: list of email addresses (optional, sends invite). "
        "google_meet: set True to auto-generate a Google Meet video link."
    ),
)
async def comm_calendar_create_event(
    summary: str,
    start: str,
    end: str,
    source: str = "apopovic",
    description: str = "",
    location: str = "",
    attendees: Optional[List[str]] = None,
    timezone: str = "Europe/Vienna",
    google_meet: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "summary": summary,
        "start": start,
        "end": end,
        "description": description,
        "timezone": timezone,
        "google_meet": google_meet,
    }
    if location:
        payload["location"] = location
    if attendees:
        payload["attendees"] = attendees
    return await call_comm_api("POST", f"/api/v1/calendar/{source}/events", json=payload)


@comm_mcp.tool(
    name="calendar_delete_event",
    description="Delete a Google Calendar event by ID.",
)
async def comm_calendar_delete_event(
    event_id: str,
    source: str = "apopovic",
) -> Dict[str, Any]:
    return await call_comm_api("DELETE", f"/api/v1/calendar/{source}/events/{event_id}")


comm_app = comm_mcp.streamable_http_app()
mount_mcp("comm", COMM_PATH, comm_app)


# ── Story Architect API ──

story_mcp = FastMCP(
    name="story",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@story_mcp.tool(
    name="projects_list",
    description="List all story projects. Optional filters: status (draft/active/archived), genre.",
)
async def story_projects_list(
    status: Optional[str] = None,
    genre: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    params = _clean_params(status=status, genre=genre, limit=limit, offset=offset)
    return await call_story_api("GET", "/api/v1/projects/", params=params)


@story_mcp.tool(
    name="projects_get",
    description="Get a story project with full hierarchy: beats > scenes > shots > media/prompts + characters/locations/artifacts.",
)
async def story_projects_get(project_id: int) -> Dict[str, Any]:
    return await call_story_api("GET", f"/api/v1/projects/{project_id}")


@story_mcp.tool(
    name="projects_create",
    description="Create a new story project. Required: title. Optional: description, genre, tagline, status, metadata_json.",
)
async def story_projects_create(
    title: str,
    description: Optional[str] = None,
    genre: Optional[str] = None,
    tagline: Optional[str] = None,
    status: str = "draft",
    metadata_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, description=description, genre=genre, tagline=tagline, status=status, metadata_json=metadata_json)
    return await call_story_api("POST", "/api/v1/projects/", json_body=body)


@story_mcp.tool(
    name="projects_update",
    description="Update a story project. All fields optional.",
)
async def story_projects_update(
    project_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    genre: Optional[str] = None,
    tagline: Optional[str] = None,
    status: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, description=description, genre=genre, tagline=tagline, status=status, metadata_json=metadata_json)
    return await call_story_api("PUT", f"/api/v1/projects/{project_id}", json_body=body)


@story_mcp.tool(
    name="projects_delete",
    description="Delete a story project and all its content (cascade).",
)
async def story_projects_delete(project_id: int) -> Dict[str, Any]:
    return await call_story_api("DELETE", f"/api/v1/projects/{project_id}")


@story_mcp.tool(
    name="beats_list",
    description="List all story beats for a project, ordered by position.",
)
async def story_beats_list(project_id: int) -> Dict[str, Any]:
    return await call_story_api("GET", f"/api/v1/projects/{project_id}/beats")


@story_mcp.tool(
    name="beats_create",
    description="Create a new story beat. Required: project_id, title. Optional: summary, dramatic_function, emotional_effect, position.",
)
async def story_beats_create(
    project_id: int,
    title: str,
    summary: Optional[str] = None,
    dramatic_function: Optional[str] = None,
    emotional_effect: Optional[str] = None,
    position: Optional[int] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, summary=summary, dramatic_function=dramatic_function, emotional_effect=emotional_effect, position=position)
    return await call_story_api("POST", f"/api/v1/projects/{project_id}/beats", json_body=body)


@story_mcp.tool(
    name="beats_update",
    description="Update a story beat. All fields optional.",
)
async def story_beats_update(
    beat_id: int,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    dramatic_function: Optional[str] = None,
    emotional_effect: Optional[str] = None,
    position: Optional[int] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, summary=summary, dramatic_function=dramatic_function, emotional_effect=emotional_effect, position=position)
    return await call_story_api("PUT", f"/api/v1/beats/{beat_id}", json_body=body)


@story_mcp.tool(
    name="scenes_list",
    description="List all scenes for a beat, ordered by position.",
)
async def story_scenes_list(beat_id: int) -> Dict[str, Any]:
    return await call_story_api("GET", f"/api/v1/beats/{beat_id}/scenes")


@story_mcp.tool(
    name="scenes_create",
    description="Create a new scene. Required: beat_id, title. Optional: description, purpose, characters, location, time_of_day, transition.",
)
async def story_scenes_create(
    beat_id: int,
    title: str,
    description: Optional[str] = None,
    purpose: Optional[str] = None,
    characters: Optional[str] = None,
    location: Optional[str] = None,
    time_of_day: Optional[str] = None,
    transition: Optional[str] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, description=description, purpose=purpose, characters=characters, location=location, time_of_day=time_of_day, transition=transition)
    return await call_story_api("POST", f"/api/v1/beats/{beat_id}/scenes", json_body=body)


@story_mcp.tool(
    name="scenes_update",
    description="Update a scene. All fields optional.",
)
async def story_scenes_update(
    scene_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    purpose: Optional[str] = None,
    characters: Optional[str] = None,
    location: Optional[str] = None,
    time_of_day: Optional[str] = None,
    transition: Optional[str] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, description=description, purpose=purpose, characters=characters, location=location, time_of_day=time_of_day, transition=transition)
    return await call_story_api("PUT", f"/api/v1/scenes/{scene_id}", json_body=body)


@story_mcp.tool(
    name="shots_list",
    description="List all shots for a scene, ordered by position.",
)
async def story_shots_list(scene_id: int) -> Dict[str, Any]:
    return await call_story_api("GET", f"/api/v1/scenes/{scene_id}/shots")


@story_mcp.tool(
    name="shots_create",
    description="Create a new shot. Required: scene_id, title. Optional: description, camera_perspective, camera_movement, mood, lighting, duration_sec, audio_hint, voice_over, production_status.",
)
async def story_shots_create(
    scene_id: int,
    title: str,
    description: Optional[str] = None,
    camera_perspective: Optional[str] = None,
    camera_movement: Optional[str] = None,
    mood: Optional[str] = None,
    lighting: Optional[str] = None,
    duration_sec: Optional[float] = None,
    audio_hint: Optional[str] = None,
    voice_over: Optional[str] = None,
    production_status: str = "planned",
) -> Dict[str, Any]:
    body = _clean_params(title=title, description=description, camera_perspective=camera_perspective, camera_movement=camera_movement, mood=mood, lighting=lighting, duration_sec=duration_sec, audio_hint=audio_hint, voice_over=voice_over, production_status=production_status)
    return await call_story_api("POST", f"/api/v1/scenes/{scene_id}/shots", json_body=body)


@story_mcp.tool(
    name="shots_update",
    description="Update a shot. All fields optional.",
)
async def story_shots_update(
    shot_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    camera_perspective: Optional[str] = None,
    camera_movement: Optional[str] = None,
    mood: Optional[str] = None,
    lighting: Optional[str] = None,
    duration_sec: Optional[float] = None,
    audio_hint: Optional[str] = None,
    voice_over: Optional[str] = None,
    production_status: Optional[str] = None,
) -> Dict[str, Any]:
    body = _clean_params(title=title, description=description, camera_perspective=camera_perspective, camera_movement=camera_movement, mood=mood, lighting=lighting, duration_sec=duration_sec, audio_hint=audio_hint, voice_over=voice_over, production_status=production_status)
    return await call_story_api("PUT", f"/api/v1/shots/{shot_id}", json_body=body)


@story_mcp.tool(
    name="shots_attach_media",
    description="Attach a media reference (from Storage API) to a shot. Roles: thumbnail, reference, variant, final, keyframe_start, keyframe_end.",
)
async def story_shots_attach_media(
    shot_id: int,
    storage_id: int,
    role: str = "reference",
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    body = _clean_params(storage_id=storage_id, role=role, caption=caption)
    return await call_story_api("POST", f"/api/v1/shots/{shot_id}/media", json_body=body)


@story_mcp.tool(
    name="prompts_create",
    description="Create a prompt for a shot. Types: image, video, motion, voiceover, music.",
)
async def story_prompts_create(
    shot_id: int,
    prompt_type: str,
    prompt_text: str,
    version: int = 1,
    is_active: bool = True,
) -> Dict[str, Any]:
    body = dict(prompt_type=prompt_type, prompt_text=prompt_text, version=version, is_active=is_active)
    return await call_story_api("POST", f"/api/v1/shots/{shot_id}/prompts", json_body=body)


@story_mcp.tool(
    name="characters_list",
    description="List all characters for a project.",
)
async def story_characters_list(project_id: int) -> Dict[str, Any]:
    return await call_story_api("GET", f"/api/v1/projects/{project_id}/characters")


@story_mcp.tool(
    name="characters_create",
    description="Create a character. Required: project_id, name. Optional: description, origin, powers, symbol, outfit, colors, behavior, mission, prompt_tokens.",
)
async def story_characters_create(
    project_id: int,
    name: str,
    description: Optional[str] = None,
    origin: Optional[str] = None,
    powers: Optional[str] = None,
    symbol: Optional[str] = None,
    outfit: Optional[str] = None,
    colors: Optional[str] = None,
    behavior: Optional[str] = None,
    mission: Optional[str] = None,
    prompt_tokens: Optional[str] = None,
) -> Dict[str, Any]:
    body = _clean_params(name=name, description=description, origin=origin, powers=powers, symbol=symbol, outfit=outfit, colors=colors, behavior=behavior, mission=mission, prompt_tokens=prompt_tokens)
    return await call_story_api("POST", f"/api/v1/projects/{project_id}/characters", json_body=body)


@story_mcp.tool(
    name="export_project",
    description="Export a full project. Format: json or markdown.",
)
async def story_export_project(
    project_id: int,
    format: str = "json",
) -> Dict[str, Any]:
    return await call_story_api("GET", f"/api/v1/projects/{project_id}/export", params={"format": format})


story_app = story_mcp.streamable_http_app()
mount_mcp("story", STORY_PATH, story_app)


# ── Cloud API (Inter-Agent Communication) ──

cloud_mcp = FastMCP(
    name="cloud-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@cloud_mcp.tool(
    name="send_message",
    description="Send a message to another Claude Code agent on any server (Mac or arkserver). Auto-routes to the correct server. Default is fire-and-forget: message is delivered to the agent's tmux, you get back {id, status: 'sent'} immediately. The other agent's response will appear in YOUR tmux pane later as an [IACP:...] marker — you'll see it next time you're active. You do NOT need to wait. Only set wait=true for rare cases where you need the response inline (short queries, <30s expected).",
)
async def cloud_send_message(
    from_session: str,
    to_session: str,
    message: str,
    timeout: int = 30,
    wait: bool = False,
) -> Dict[str, Any]:
    return await call_cloud_api(
        "POST", "/api/agents/route",
        json_body={"from": from_session, "to": to_session, "message": message, "wait": wait, "timeout": timeout},
    )


@cloud_mcp.tool(
    name="list_agents",
    description="List all active agents across all servers (Mac and arkserver). Shows agent name, server location, and status.",
)
async def cloud_list_agents() -> Dict[str, Any]:
    return await call_cloud_api("GET", "/api/agents/registry")


@cloud_mcp.tool(
    name="read_session",
    description="Read the current screen content of any agent's tmux session across all servers. Returns plain text, no ANSI codes.",
)
async def cloud_read_session(session_name: str, lines: int = 50) -> Dict[str, Any]:
    return await call_cloud_api("GET", f"/api/agents/read/{session_name}", params={"lines": lines})


@cloud_mcp.tool(
    name="inbox",
    description="Check your inbox for messages from other agents.",
)
async def cloud_inbox(session_name: str) -> Dict[str, Any]:
    return await call_cloud_api("GET", f"/api/agents/inbox/{session_name}")


@cloud_mcp.tool(
    name="create_session",
    description=(
        "Create a new agent session, federation-aware. Returns {status, name, node}. "
        "Use agent='claude' (default), 'codex', or 'gemini'. "
        "Set node='arkserver|arkturian|oneal|pdrei|mac' to spawn on a specific federation peer "
        "(default: local node where the cloud-api receiving this MCP call runs). "
        "owner_email assigns ownership for RBAC visibility/quota. "
        "Set pretty=True for JSON chat mode (queue/widget agents)."
    ),
)
async def cloud_create_session(
    name: str,
    pretty: bool = False,
    agent: str = "claude",
    node: str = "",
    owner_email: str = "",
    department: str = "engineering",
    role: str = "developer",
    auto_restart: bool = True,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "name": name,
        "pretty": pretty,
        "agent": agent,
        "auto_restart": auto_restart,
        "department": department,
        "role": role,
    }
    if node:
        body["node"] = node.lower()
    if owner_email:
        body["owner_email"] = owner_email
    return await call_cloud_api("POST", "/api/sessions", json_body=body)


@cloud_mcp.tool(
    name="self_restart",
    description=(
        "Restart your own session. Use this as a last-resort self-heal when "
        "your MCP-connection state is stuck — typical symptoms: MCP tool calls "
        "fail with 404/401 even though the underlying service is reachable via "
        "direct HTTP, '/mcp' refresh doesn't recover, you have already retried. "
        "The cloud-api kills your tmux process and starts a fresh one with "
        "--continue, so your conversation history is preserved. "
        "prepare_bot_home() runs on restart and regenerates mcp-auth-helper "
        "and .claude.json from the current host-canonical config — so any "
        "upstream URL change (like an MCP server moving to a new vhost) is "
        "picked up automatically. "
        "Federation-aware: forwards to the owning node. You must pass your "
        "own session_name (from your CLAUDE.md identity block)."
    ),
)
async def cloud_self_restart(session_name: str) -> Dict[str, Any]:
    return await call_cloud_api("POST", f"/api/sessions/{session_name}/restart")


@cloud_mcp.tool(
    name="health",
    description="Check if the Cloud API is running.",
)
async def cloud_health() -> Dict[str, Any]:
    return await call_cloud_api("GET", "/api/health")


@cloud_mcp.tool(
    name="list_roles",
    description=(
        "List IACP roles available across the cloud-api federation. "
        "Returns {role, node, session, running, owner} for every public session "
        "on every reachable node. Use this to discover who fulfills the "
        "'main-architect' role before calling send_to_role."
    ),
)
async def cloud_list_roles() -> Dict[str, Any]:
    return await call_cloud_api("GET", "/api/iacp/roles")


@cloud_mcp.tool(
    name="send_to_role",
    description=(
        "Send an IACP message to whichever cloud-api session currently fills "
        "a role (e.g. 'main-architect'). The local cloud-api resolves the role "
        "via /api/public-sessions across the federation, forwards the message "
        "to the owning node, and queues it on that node's session. "
        "Fire-and-forget: returns {id, status, node, session}. "
        "Use this when you (an agent on a remote node) need help from the "
        "main system architect — bug reports, infrastructure issues, "
        "questions about cross-system behavior."
    ),
)
async def cloud_send_to_role(
    role: str,
    text: str,
    user_id: str = "iacp",
) -> Dict[str, Any]:
    if not IACP_TOKEN:
        return {"error": "IACP_TOKEN env not configured on this MCP server"}
    return await _fetch_json(
        "POST",
        f"{CLOUD_API_BASE}/api/iacp/send/{role}",
        headers={"X-IACP-Token": IACP_TOKEN},
        json_body={"text": text, "user_id": user_id},
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Pool Management
# ---------------------------------------------------------------------------

@cloud_mcp.tool(
    name="list_pools",
    description=(
        "List all session pools across the federation. Shows pool name, size, agent, model, "
        "effort, idle timeout, and per-instance status (running, available, acquired). "
        "Pools hold N pre-warmed bot instances with acquire/release lifecycle."
    ),
)
async def cloud_list_pools() -> Dict[str, Any]:
    return await call_cloud_api("GET", "/api/pools/all")


@cloud_mcp.tool(
    name="create_pool",
    description=(
        "Create a new session pool with N pre-warmed bot instances. "
        "Instances are ephemeral: auto-cleanup on disconnect, history wiped on release. "
        "The watchdog starts instances automatically within 30 seconds. "
        "Example: create_pool('guide', size=3, agent='claude', model='sonnet', effort='low')"
    ),
)
async def cloud_create_pool(
    name: str,
    size: int = 3,
    agent: str = "claude",
    model: str = "",
    effort: str = "high",
    idle_timeout_minutes: int = 30,
) -> Dict[str, Any]:
    return await call_cloud_api(
        "POST", "/api/pools",
        json_body={
            "name": name, "size": size, "agent": agent,
            "model": model, "effort": effort,
            "idle_timeout_minutes": idle_timeout_minutes,
        },
    )


@cloud_mcp.tool(
    name="delete_pool",
    description="Delete a session pool and kill all its instances.",
)
async def cloud_delete_pool(name: str) -> Dict[str, Any]:
    return await call_cloud_api("DELETE", f"/api/pools/{name}")


@cloud_mcp.tool(
    name="pool_config",
    description=(
        "Update pool configuration. Changeable fields: size (number of instances), "
        "idle_timeout_minutes (auto-release acquired instances after N minutes idle)."
    ),
)
async def cloud_pool_config(
    name: str,
    size: int | None = None,
    idle_timeout_minutes: int | None = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if size is not None:
        body["size"] = size
    if idle_timeout_minutes is not None:
        body["idle_timeout_minutes"] = idle_timeout_minutes
    return await call_cloud_api("PATCH", f"/api/pools/{name}/config", json_body=body)


@cloud_mcp.tool(
    name="acquire_pool",
    description=(
        "Acquire a free instance from a pool for a user. Returns the instance name. "
        "The instance is marked as in-use and won't be given to another caller. "
        "Auto-releases after idle_timeout_minutes or WebSocket disconnect + grace period."
    ),
)
async def cloud_acquire_pool(
    pool_name: str,
    user_id: str = "",
) -> Dict[str, Any]:
    return await call_cloud_api(
        "POST", f"/api/pools/{pool_name}/acquire",
        json_body={"user_id": user_id},
    )


@cloud_mcp.tool(
    name="release_pool",
    description=(
        "Release an acquired pool instance. Kills the session, wipes conversation history, "
        "and sets status to registered. The watchdog will start a fresh instance within 30s."
    ),
)
async def cloud_release_pool(
    pool_name: str,
    instance_name: str,
) -> Dict[str, Any]:
    return await call_cloud_api("POST", f"/api/pools/{pool_name}/release/{instance_name}")


# ---------------------------------------------------------------------------
# Session Lifecycle
# ---------------------------------------------------------------------------

@cloud_mcp.tool(
    name="session_lifecycle",
    description=(
        "Get lifecycle settings for a session: ephemeral flag, WS grace period, "
        "idle timeout, wipe-on-cleanup, auto-restart, current WebSocket status, "
        "pool membership, and acquisition state."
    ),
)
async def cloud_session_lifecycle(session_name: str) -> Dict[str, Any]:
    return await call_cloud_api("GET", f"/api/sessions/{session_name}/lifecycle")


@cloud_mcp.tool(
    name="session_config",
    description=(
        "Update session lifecycle configuration. Fields: ephemeral (bool), "
        "ws_grace_seconds (int, delay after WS disconnect before cleanup), "
        "wipe_on_cleanup (bool, delete history on cleanup), "
        "max_idle_minutes (int, 0=never), auto_restart (bool), "
        "effort (str: high/low), model (str)."
    ),
)
async def cloud_session_config(
    session_name: str,
    ephemeral: bool | None = None,
    ws_grace_seconds: int | None = None,
    wipe_on_cleanup: bool | None = None,
    max_idle_minutes: int | None = None,
    auto_restart: bool | None = None,
    effort: str | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if ephemeral is not None:
        body["ephemeral"] = ephemeral
    if ws_grace_seconds is not None:
        body["ws_grace_seconds"] = ws_grace_seconds
    if wipe_on_cleanup is not None:
        body["wipe_on_cleanup"] = wipe_on_cleanup
    if max_idle_minutes is not None:
        body["max_idle_minutes"] = max_idle_minutes
    if auto_restart is not None:
        body["auto_restart"] = auto_restart
    if effort is not None:
        body["effort"] = effort
    if model is not None:
        body["model"] = model
    return await call_cloud_api("PATCH", f"/api/sessions/{session_name}/config", json_body=body)


@cloud_mcp.tool(
    name="system_memory",
    description=(
        "Get system memory info: total/used/free RAM, swap, bot RAM usage, "
        "max_concurrent sessions, running count, and whether a new session can start."
    ),
)
async def cloud_system_memory() -> Dict[str, Any]:
    return await call_cloud_api("GET", "/api/system/memory")


cloud_app = cloud_mcp.streamable_http_app()
mount_mcp("cloud", CLOUD_PATH, cloud_app)


# ═══════════════════════════════════════════════════════════════════
# Conversation API — threads, events, persons, agent actions
# ═══════════════════════════════════════════════════════════════════

conversation_mcp = FastMCP(
    name="conversation-api",
    streamable_http_path="/",
    stateless_http=True,
    auth=None,
    log_level="INFO",
)


@conversation_mcp.tool(
    name="list",
    description="""List conversations (threaded email/messaging threads).

    Returns most recent real (non-noise) conversations by default. Each
    conversation has: id, topic_summary, topic_clean, status, ball_with,
    last_activity_at, participants[], event_count, open_item_count,
    auto_close_decision, auto_close_confidence.

    Parameters:
    - status (optional): filter — open | waiting_me | waiting_them | stalled | closed
    - include_noise (default False): also include newsletters/notifications
      that the heuristic classifier marked as non-conversation.
    - limit (default 50): maximum number of conversations.

    Use this to scan the user's ongoing threads. Follow up with `get` on a
    specific conversation_id for the full event body.
    """,
)
async def conversation_list(
    status: Optional[str] = None,
    include_noise: bool = False,
    limit: int = 50,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    if include_noise:
        params["include_noise"] = "true"
    return await call_conversation_api("GET", "/api/v1/conversations", params=params)


@conversation_mcp.tool(
    name="get",
    description="""Get a single conversation with full details.

    Returns: id, topic_summary, topic_clean, status, ball_with,
    opened_at, last_activity_at, closed_at, is_real_conversation,
    auto_close_decision/confidence/reasoning, participants[], events[],
    open_items[].

    Parameters:
    - conversation_id (required): e.g. 'c_abc123def456'
    - include_body (default True): include each event's full content_full
      (the raw email body). Set False for a lightweight metadata-only view.

    Events come chronological, newest at the end. Each event has direction
    (in/out), channel, source_account, subject, summary, content_full (when
    included), sender (name+org), and LLM-derived flags addressed_to_me /
    requires_response / llm_intent.
    """,
)
async def conversation_get(
    conversation_id: str,
    include_body: bool = True,
) -> Dict[str, Any]:
    params = {"include_body": "true" if include_body else "false"}
    return await call_conversation_api(
        "GET", f"/api/v1/conversations/{conversation_id}", params=params,
    )


@conversation_mcp.tool(
    name="event",
    description="""Get a single event (message) with its full body.

    Useful when you only need one specific email out of a large thread and
    want to skip loading the rest. Returns: id, conversation_id, timestamp,
    direction (in/out), channel, source_account, subject, summary,
    content_full, raw_from, raw_to, sender_name, sender_org,
    message_type (conversation|notification|marketing|etc.),
    addressed_to_me, requires_response, llm_intent, labels,
    external_message_id, external_thread_id.

    Parameters:
    - conversation_id: parent conversation id
    - event_id: the event (e.g. 'e_abc123def456')
    """,
)
async def conversation_event(
    conversation_id: str, event_id: str,
) -> Dict[str, Any]:
    return await call_conversation_api(
        "GET", f"/api/v1/conversations/{conversation_id}/events/{event_id}",
    )


@conversation_mcp.tool(
    name="timeline",
    description="""Return conversations + their events in a compact timeline shape.

    Optimised for chronological overview / visualisation; each conversation
    carries its event list flat. Parameters mirror `list`:
    - status (optional)
    - channel (optional): filter events by channel (gmail|imap|telegram|...)
    - min_events (default 1): skip conversations with fewer events
    - include_noise (default False)
    - limit (default 200)

    Also returns `time_range` (min/max timestamps across all events returned).
    """,
)
async def conversation_timeline(
    status: Optional[str] = None,
    channel: Optional[str] = None,
    min_events: int = 1,
    include_noise: bool = False,
    limit: int = 200,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"min_events": min_events, "limit": limit}
    if status:
        params["status"] = status
    if channel:
        params["channel"] = channel
    if include_noise:
        params["include_noise"] = "true"
    return await call_conversation_api("GET", "/api/v1/timeline", params=params)


@conversation_mcp.tool(
    name="graph",
    description="""Return the person-relationship graph backing the force-directed view.

    Nodes are persons (one synthetic 'me' plus all peers), links are
    conversations (one edge per conversation between me and the peer, with
    topic, event_count, status, ball_with).

    Parameters:
    - status (optional): filter by conversation status
    - include_noise (default False)

    Use when you want to reason about who the user communicates with most
    intensively and on which topics.
    """,
)
async def conversation_graph(
    status: Optional[str] = None,
    include_noise: bool = False,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if status:
        params["status"] = status
    if include_noise:
        params["include_noise"] = "true"
    return await call_conversation_api("GET", "/api/v1/graph", params=params)


@conversation_mcp.tool(
    name="persons_list",
    description="""List all known persons across conversations.

    Each person has: id, canonical_name, org, role, is_owner, id_count,
    event_count, conv_count, identifiers[] (channel+value pairs covering
    email and — later — telegram/whatsapp).

    Owner (the user themselves) appears first, then peers sorted by
    conversation count descending.
    """,
)
async def conversation_persons_list() -> Dict[str, Any]:
    return await call_conversation_api("GET", "/api/v1/persons")


@conversation_mcp.tool(
    name="person_get",
    description="""Get one person with their identifiers and counts.

    Parameters:
    - person_id (required): e.g. 'p_abc123def456'
    """,
)
async def conversation_person_get(person_id: str) -> Dict[str, Any]:
    return await call_conversation_api("GET", f"/api/v1/persons/{person_id}")


@conversation_mcp.tool(
    name="person_threads",
    description="""All conversations involving one peer, with their events + participants.

    Returns {person, conversations: [...]}  where each conversation has
    events[], participants[], and status/ball_with metadata.

    Parameters:
    - person_id (required)

    Use this when the user says 'show me everything I have with Thomas' —
    single call gets the full per-person picture.
    """,
)
async def conversation_person_threads(person_id: str) -> Dict[str, Any]:
    return await call_conversation_api(
        "GET", f"/api/v1/persons/{person_id}/threads",
    )


@conversation_mcp.tool(
    name="actions_list",
    description="""List agent actions (dispatches) attached to a conversation.

    Each action: id, agent_name, user_prompt, status (pending|sent|done|
    error|accepted|dismissed), response, error_msg, created_at, resolved_at,
    parent_action_id (for threaded replies).

    Parameters:
    - conversation_id (required)
    """,
)
async def conversation_actions_list(conversation_id: str) -> Dict[str, Any]:
    return await call_conversation_api(
        "GET", f"/api/v1/conversations/{conversation_id}/actions",
    )


@conversation_mcp.tool(
    name="dispatch",
    description="""Dispatch an instruction to an agent from within a conversation context.

    Creates an agent_action row, posts to cloud-api's /api/queue with the
    full conversation context bundle (topic, open items, recent events,
    latest event body), and returns the queue_id. The agent processes async;
    poll `actions_list` or `get` to see the response land in the conversation.

    Parameters:
    - conversation_id (required): conversation to act on
    - agent_name (required, unless parent_action_id): target agent session name
    - prompt (required): what to tell the agent (e.g. 'draft a reply declining politely')
    - parent_action_id (optional): integer id of an existing action — if set,
      this is a threaded REPLY continuation; agent_name is inherited from the
      parent if omitted, and the context bundle is trimmed (agent already has
      prior context in its session).

    Returns: {action_id, agent_name, parent_action_id, send_result: {ok, queue_id, status, attempts}}
    """,
)
async def conversation_dispatch(
    conversation_id: str,
    prompt: str,
    agent_name: Optional[str] = None,
    parent_action_id: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": prompt}
    if agent_name:
        payload["agent_name"] = agent_name
    if parent_action_id is not None:
        payload["parent_action_id"] = parent_action_id
    return await call_conversation_api(
        "POST",
        f"/api/v1/conversations/{conversation_id}/dispatch",
        json_body=payload,
    )


@conversation_mcp.tool(
    name="action_accept",
    description="""Mark an agent action as accepted (user is happy with the response).

    Parameters:
    - action_id (required): integer id of the agent_action
    """,
)
async def conversation_action_accept(action_id: int) -> Dict[str, Any]:
    return await call_conversation_api(
        "POST", f"/api/v1/agent_actions/{action_id}/accept",
    )


@conversation_mcp.tool(
    name="action_dismiss",
    description="""Mark an agent action as dismissed (ignore the response, don't act on it).

    Parameters:
    - action_id (required)
    """,
)
async def conversation_action_dismiss(action_id: int) -> Dict[str, Any]:
    return await call_conversation_api(
        "POST", f"/api/v1/agent_actions/{action_id}/dismiss",
    )


conversation_app = conversation_mcp.streamable_http_app()
mount_mcp("conversation", CONVERSATION_PATH, conversation_app)


_storage_stack = AsyncExitStack()
_oneal_stack = AsyncExitStack()
_oneal_storage_stack = AsyncExitStack()
_artrack_stack = AsyncExitStack()
_codepilot_stack = AsyncExitStack()
_content_stack = AsyncExitStack()
_tree_stack = AsyncExitStack()
_ai_stack = AsyncExitStack()
_tarot_stack = AsyncExitStack()
_business_stack = AsyncExitStack()
_comm_stack = AsyncExitStack()
_knowledge_stack = AsyncExitStack()
_review_stack = AsyncExitStack()
_story_stack = AsyncExitStack()
_cloud_stack = AsyncExitStack()
_conversation_stack = AsyncExitStack()


# Track which MCP sessions were started (for shutdown ordering)
_started_mcps: List[Any] = []


def _all_mcps():
    """All MCP definitions in mount order: (name, stack, mcp_instance)."""
    return [
        ("storage", _storage_stack, storage_mcp),
        ("oneal", _oneal_stack, oneal_mcp),
        ("oneal-storage", _oneal_storage_stack, oneal_storage_mcp),
        ("artrack", _artrack_stack, artrack_mcp),
        ("codepilot", _codepilot_stack, codepilot_mcp),
        ("content", _content_stack, content_mcp),
        ("tree", _tree_stack, tree_mcp),
        ("ai", _ai_stack, ai_mcp),
        ("tarot", _tarot_stack, tarot_mcp),
        ("business", _business_stack, business_mcp),
        ("comm", _comm_stack, comm_mcp),
        ("knowledge", _knowledge_stack, knowledge_mcp),
        ("review", _review_stack, review_mcp),
        ("story", _story_stack, story_mcp),
        ("cloud", _cloud_stack, cloud_mcp),
        ("conversation", _conversation_stack, conversation_mcp),
    ]


@app.on_event("startup")
async def startup() -> None:
    """Initialize session managers for all enabled MCPs.

    Compatible with both old (0.27.x) and new (1.x) Starlette versions.
    Respects MCP_SERVERS filter — only initializes enabled MCPs.
    """
    for name, stack, mcp in _all_mcps():
        if ENABLED_MCPS is not None and name not in ENABLED_MCPS:
            continue
        await stack.enter_async_context(mcp.session_manager.run())
        _started_mcps.append((name, stack))
        logger.info("Started MCP session: %s", name)


@app.on_event("shutdown")
async def shutdown() -> None:
    """Shut down session managers in reverse order."""
    for name, stack in reversed(_started_mcps):
        try:
            await stack.aclose()
            logger.info("Stopped MCP session: %s", name)
        except Exception as e:
            logger.warning("Error stopping MCP %s: %s", name, e)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Human-friendly service descriptor."""
    return {
        "name": "arkturian-mcp",
        "version": "3.0.0",
        "description": "Arkturian MCP Aggregator with per-tenant isolation, human-in-the-loop, AI generation, Content API, Business API, Knowledge API, and Tarot",
        "endpoints": {
            "storage": {
                "path": STORAGE_PATH,
                "tenant": "arkturian",
                "tools": [tool.name for tool in storage_mcp._tool_manager.list_tools()],
                "upstream": STORAGE_API_BASE,
            },
            "oneal": {
                "path": ONEAL_PATH,
                "tools": [tool.name for tool in oneal_mcp._tool_manager.list_tools()],
                "upstream": ONEAL_API_BASE,
            },
            "oneal-storage": {
                "path": ONEAL_STORAGE_PATH,
                "tenant": "oneal",
                "tools": [tool.name for tool in oneal_storage_mcp._tool_manager.list_tools()],
                "upstream": STORAGE_API_BASE,
            },
            "artrack": {
                "path": ARTRACK_PATH,
                "tools": [tool.name for tool in artrack_mcp._tool_manager.list_tools()],
                "upstream": ARTRACK_API_BASE,
            },
            "codepilot": {
                "path": CODEPILOT_PATH,
                "tools": [tool.name for tool in codepilot_mcp._tool_manager.list_tools()],
                "upstream": COMM_API_BASE,
                "description": "Human-in-the-loop tools for CodePilot",
            },
            "ai": {
                "path": AI_PATH,
                "tools": [tool.name for tool in ai_mcp._tool_manager.list_tools()],
                "upstream": AI_API_BASE,
                "description": "AI text, vision, and image generation tools",
            },
            "content": {
                "path": CONTENT_PATH,
                "tools": [tool.name for tool in content_mcp._tool_manager.list_tools()],
                "upstream": CONTENT_API_BASE,
                "description": "Content management API for posts, media, annotations, and blocks",
            },
            "tree": {
                "path": TREE_PATH,
                "tools": [tool.name for tool in tree_mcp._tool_manager.list_tools()],
                "upstream": TREE_API_BASE,
                "description": "Collaborative tree editing with node-level CRUD and real-time sync",
            },
            "tarot": {
                "path": TAROT_PATH,
                "tools": [tool.name for tool in tarot_mcp._tool_manager.list_tools()],
                "upstream": TOOLS_API_BASE,
                "description": "Tarot card reading with full 78-card deck",
            },
            "business": {
                "path": BUSINESS_PATH,
                "tools": [tool.name for tool in business_mcp._tool_manager.list_tools()],
                "upstream": BUSINESS_API_BASE,
                "description": "Business management: Honorarnoten, Rechnungen, Kunden, Transaktionen, Dashboard",
            },
            "comm": {
                "path": COMM_PATH,
                "tools": [tool.name for tool in comm_mcp._tool_manager.list_tools()],
                "upstream": COMM_API_BASE,
                "description": "Unified communication: Email, Telegram, Interventions",
            },
            "knowledge": {
                "path": KNOWLEDGE_PATH,
                "tools": [tool.name for tool in knowledge_mcp._tool_manager.list_tools()],
                "upstream": KNOWLEDGE_API_BASE,
                "description": "AI-powered knowledge extraction, annotated knowledge with visual annotations, and audio generation",
            },
            "review": {
                "path": REVIEW_PATH,
                "tools": [tool.name for tool in review_mcp._tool_manager.list_tools()],
                "upstream": REVIEW_API_BASE,
                "description": "AI-powered multi-perspective review orchestrator for websites, documents, presentations, and content",
            },
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Aggregated health check for all upstream services."""
    results: Dict[str, Any] = {"status": "healthy"}

    try:
        results["storage_arkturian"] = await storage_kg_stats()
    except httpx.HTTPError as exc:
        results["status"] = "degraded"
        results["storage_arkturian_error"] = str(exc)

    try:
        results["oneal_products"] = await oneal_service_ping()
    except httpx.HTTPError as exc:
        results["status"] = "degraded"
        results["oneal_products_error"] = str(exc)

    try:
        results["storage_oneal"] = await oneal_storage_kg_stats()
    except httpx.HTTPError as exc:
        results["status"] = "degraded"
        results["storage_oneal_error"] = str(exc)

    try:
        results["artrack"] = await artrack_service_health()
    except httpx.HTTPError as exc:
        results["status"] = "degraded"
        results["artrack_error"] = str(exc)

    try:
        results["content"] = await content_service_health()
    except httpx.HTTPError as exc:
        results["status"] = "degraded"
        results["content_error"] = str(exc)

    try:
        results["tree"] = await tree_service_health()
    except httpx.HTTPError as exc:
        results["status"] = "degraded"
        results["tree_error"] = str(exc)

    if BUSINESS_API_KEY:
        try:
            results["business"] = await business_service_health()
        except httpx.HTTPError as exc:
            results["status"] = "degraded"
            results["business_error"] = str(exc)

    if COMM_API_KEY:
        try:
            results["comm"] = await comm_service_health()
        except httpx.HTTPError as exc:
            results["status"] = "degraded"
            results["comm_error"] = str(exc)

    if results["status"] != "healthy":
        raise HTTPException(status_code=207, detail=results)
    return results


@app.get("/.well-known/mcp.json")
async def well_known(request: Request) -> JSONResponse:
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    return JSONResponse(
        {
            "mcpServers": {
                "storage": {
                    "name": "arkturian-storage",
                    "version": "2.6.0",
                    "tenant": "arkturian",
                    "url": f"{base_url}{STORAGE_PATH}/",
                },
                "oneal": {
                    "name": "oneal-products",
                    "version": "1.0.0",
                    "url": f"{base_url}{ONEAL_PATH}/",
                },
                "oneal-storage": {
                    "name": "oneal-storage",
                    "version": "1.0.0",
                    "tenant": "oneal",
                    "url": f"{base_url}{ONEAL_STORAGE_PATH}/",
                },
                "artrack": {
                    "name": "artrack-api",
                    "version": "1.0.0",
                    "url": f"{base_url}{ARTRACK_PATH}/",
                },
                "codepilot": {
                    "name": "codepilot-human",
                    "version": "1.0.0",
                    "description": "Human-in-the-loop tools for CodePilot",
                    "url": f"{base_url}{CODEPILOT_PATH}/",
                },
                "ai": {
                    "name": "ai-api",
                    "version": "1.0.0",
                    "description": "AI text, vision, and image generation tools",
                    "url": f"{base_url}{AI_PATH}/",
                },
                "content": {
                    "name": "content-api",
                    "version": "1.0.0",
                    "description": "Content management for posts, media, annotations, and blocks",
                    "url": f"{base_url}{CONTENT_PATH}/",
                },
                "tree": {
                    "name": "tree-api",
                    "version": "1.0.0",
                    "description": "Collaborative tree editing with node-level CRUD",
                    "url": f"{base_url}{TREE_PATH}/",
                },
                "tarot": {
                    "name": "tarot-api",
                    "version": "1.0.0",
                    "description": "Tarot card reading with full 78-card deck",
                    "url": f"{base_url}{TAROT_PATH}/",
                },
                "business": {
                    "name": "business-api",
                    "version": "1.0.0",
                    "description": "Business management: Honorarnoten, invoices, clients, transactions, dashboard",
                    "url": f"{base_url}{BUSINESS_PATH}/",
                },
                "comm": {
                    "name": "comm-api",
                    "version": "1.0.0",
                    "description": "Unified communication: Email, Telegram, Interventions",
                    "url": f"{base_url}{COMM_PATH}/",
                },
                "knowledge": {
                    "name": "knowledge-api",
                    "version": "1.0.0",
                    "description": "AI-powered knowledge extraction, annotations, and audio",
                    "url": f"{base_url}{KNOWLEDGE_PATH}/",
                },
            }
        }
    )


def run() -> None:
    logger.info("Starting Arkturian MCP server on %s:%s", HOST, PORT)
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
