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
"""

from __future__ import annotations

import logging
import os
import time
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

ONEAL_API_BASE = os.getenv("ONEAL_API_BASE", "https://oneal-api.arkturian.com")
ONEAL_API_KEY = os.getenv("ONEAL_API_KEY", "oneal_demo_token")

# O'Neal Storage API (same base URL as Arkturian, different API key for tenant isolation)
ONEAL_STORAGE_API_KEY = os.getenv("ONEAL_STORAGE_API_KEY", "")

# Artrack API
ARTRACK_API_BASE = os.getenv("ARTRACK_API_BASE", "https://api-artrack.arkturian.com")
ARTRACK_API_KEY = os.getenv("ARTRACK_API_KEY", "")

# Content API
CONTENT_API_BASE = os.getenv("CONTENT_API_BASE", "https://content-api.arkturian.com")

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

if not STORAGE_API_KEY:
    raise RuntimeError("ARKTURIAN_API_KEY environment variable must be set.")
if not ONEAL_STORAGE_API_KEY:
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

# Tools API (for Tarot)
TOOLS_API_BASE = os.getenv("TOOLS_API_BASE", "https://tools-api.arkturian.com")

# Business API
BUSINESS_API_BASE = os.getenv("BUSINESS_API_BASE", "https://business-api.arkturian.com")
BUSINESS_API_KEY = os.getenv("BUSINESS_API_KEY", "")

# Comm API
COMM_API_BASE = os.getenv("COMM_API_BASE", "https://comm-api.arkturian.com")
COMM_API_KEY = os.getenv("COMM_API_KEY", "")

# Knowledge API
KNOWLEDGE_API_BASE = os.getenv("KNOWLEDGE_API_BASE", "https://knowledge-api.arkturian.com")

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
            logger.error("HTTP error %s %s: %s", method, url, exc.response.text)
            raise
        except httpx.HTTPError as exc:
            logger.error("Request to %s failed: %s", url, exc)
            raise


async def call_storage_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    return await _fetch_json(
        method,
        f"{STORAGE_API_BASE}{endpoint}",
        headers={"X-API-KEY": STORAGE_API_KEY},
        params=params,
        json_body=json_body,
    )


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


async def call_content_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Content API for posts, media, annotations, and blocks."""
    return await _fetch_json(
        method,
        f"{CONTENT_API_BASE}{endpoint}",
        headers={},  # No auth required for public endpoints
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
    """Call Tree API for projects, nodes, and tree operations."""
    return await _fetch_json(
        method,
        f"{TREE_API_BASE}{endpoint}",
        headers={},
        params=params,
        json_body=json_body,
    )


async def call_business_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call Business API for invoicing, clients, transactions, and documents."""
    return await _fetch_json(
        method,
        f"{BUSINESS_API_BASE}{endpoint}",
        headers={"X-API-Key": BUSINESS_API_KEY},
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
    description="List storage objects with optional filters.",
)
async def storage_assets_list(
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
    description="""Load a media asset and return it as a Base64 Data URL.

    This is useful for displaying images in environments with Content Security Policy restrictions
    (like Claude Web Artifacts) that don't allow external image URLs.

    Returns a data URL like: data:image/png;base64,iVBORw0KGgo...
    """,
)
async def storage_media_as_data_url(
    id: int,  # noqa: A002
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Dict[str, Any]:
    import base64

    # Build URL with optional parameters
    options = _clean_params(
        width=width,
        height=height,
        format=format,
        quality=quality,
    )
    url = httpx.URL(f"{STORAGE_API_BASE}/storage/media/{id}").copy_with(params=options)

    # Fetch the image
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        response = await client.get(str(url))
        response.raise_for_status()

        # Get content type
        content_type = response.headers.get("content-type", "image/png")

        # Convert to base64
        image_bytes = response.content
        base64_data = base64.b64encode(image_bytes).decode('utf-8')

        # Create data URL
        data_url = f"data:{content_type};base64,{base64_data}"

        return {
            "data_url": data_url,
            "content_type": content_type,
            "size_bytes": len(image_bytes),
            "size_kb": round(len(image_bytes) / 1024, 2),
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

    Use storage.media_url + transformation parameters to get optimized images.
    See products_get for full transformation parameter documentation.
    """,
)
async def oneal_products_list(
    search: Optional[str] = None,
    category: Optional[str] = None,
    season: Optional[int] = None,
    cert: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    sort: Optional[str] = None,
    order: str = "asc",
    limit: int = 50,
    offset: int = 0,
    format: Optional[str] = None,
) -> Dict[str, Any]:
    params = _clean_params(
        search=search,
        limit=limit,
        offset=offset,
    )
    return await call_storage_api("GET", "/storage/oneal/products", params=params)


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
    return await call_storage_api("GET", f"/storage/oneal/products/{product_id}")


@oneal_mcp.tool(
    name="facets_list",
    description="List product facets (categories, certifications, etc.).",
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
    description="""Load an O'Neal media asset and return it as a Base64 Data URL.

    This is useful for displaying images in environments with Content Security Policy restrictions
    (like Claude Web Artifacts) that don't allow external image URLs.

    Returns a data URL like: data:image/png;base64,iVBORw0KGgo...
    """,
)
async def oneal_storage_media_as_data_url(
    id: int,  # noqa: A002
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    quality: Optional[int] = None,
) -> Dict[str, Any]:
    import base64

    # Build URL with optional parameters
    options = _clean_params(
        width=width,
        height=height,
        format=format,
        quality=quality,
    )
    url = httpx.URL(f"{STORAGE_API_BASE}/storage/media/{id}").copy_with(params=options)

    # Fetch the image
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        response = await client.get(str(url))
        response.raise_for_status()

        # Get content type
        content_type = response.headers.get("content-type", "image/png")

        # Convert to base64
        image_bytes = response.content
        base64_data = base64.b64encode(image_bytes).decode('utf-8')

        # Create data URL
        data_url = f"data:{content_type};base64,{base64_data}"

        return {
            "data_url": data_url,
            "content_type": content_type,
            "size_bytes": len(image_bytes),
            "size_kb": round(len(image_bytes) / 1024, 2),
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
    return await call_artrack_api("GET", "/tracks/")


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
async def codepilot_notify_human(message: str) -> Dict[str, Any]:
    """Send notification to human via Telegram (routed through Comm API)."""
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "sent": False}

    try:
        result = await call_comm_api(
            "POST",
            "/api/v1/telegram/interventions/notification",
            json_body={"message": message},
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
) -> Dict[str, Any]:
    """Ask human a question and wait for response (routed through Comm API)."""
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "response": None}

    try:
        # Determine endpoint based on options
        if options and len(options) > 0:
            # Approval request with buttons
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/approval",
                json_body={
                    "message": question,
                    "options": options,
                    "timeout_seconds": timeout_seconds,
                },
            )
        else:
            # Text input request
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/text-input",
                json_body={
                    "message": question,
                    "timeout_seconds": timeout_seconds,
                },
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

    Filters: status (draft|published|archived), author_id, content_type
    """,
)
async def content_posts_list(
    status: Optional[str] = None,
    author_id: Optional[str] = None,
    content_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    params = _clean_params(
        status=status,
        author_id=author_id,
        content_type=content_type,
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
    name="posts_delete",
    description="Delete a post by ID. Returns success status.",
)
async def content_posts_delete(post_id: int) -> Dict[str, Any]:
    return await call_content_api("DELETE", f"/api/v1/posts/{post_id}")


@content_mcp.tool(
    name="media_list",
    description="""List media attachments for a post.

    Returns all media items attached to the specified post with:
    - id, storage_id, position, caption, role
    - media_url: Direct URL to the media file
    - created_at
    """,
)
async def content_media_list(post_id: int) -> Dict[str, Any]:
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
async def content_annotations_list(post_id: int) -> Dict[str, Any]:
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
async def content_blocks_list(post_id: int) -> Dict[str, Any]:
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
async def content_references_list(post_id: int) -> Dict[str, Any]:
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

app = FastAPI(title="arkturian-mcp", version="2.9.0", description="Arkturian MCP Aggregator with AI generation, Content API, Tree API, Business API, and Comm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount MCP transports
storage_app = storage_mcp.streamable_http_app()
oneal_app = oneal_mcp.streamable_http_app()
oneal_storage_app = oneal_storage_mcp.streamable_http_app()
artrack_app = artrack_mcp.streamable_http_app()
codepilot_app = codepilot_mcp.streamable_http_app()
content_app = content_mcp.streamable_http_app()
tree_app = tree_mcp.streamable_http_app()
app.mount(STORAGE_PATH, storage_app)
app.mount(ONEAL_PATH, oneal_app)
app.mount(ONEAL_STORAGE_PATH, oneal_storage_app)
app.mount(ARTRACK_PATH, artrack_app)
app.mount(CODEPILOT_PATH, codepilot_app)
app.mount(CONTENT_PATH, content_app)
app.mount(TREE_PATH, tree_app)
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
app.mount(AI_PATH, ai_app)

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
app.mount(KNOWLEDGE_PATH, knowledge_app)

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
app.mount(TAROT_PATH, tarot_app)

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
async def business_dashboard_summary() -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/dashboard/summary")


@business_mcp.tool(
    name="dashboard_cashflow",
    description="Get monthly cashflow data (income, expense, profit per month) for a given year.",
)
async def business_dashboard_cashflow(year: Optional[int] = None) -> List[Dict[str, Any]]:
    params = _clean_params(year=year)
    return await call_business_api("GET", "/api/v1/dashboard/cashflow", params=params)


# --- Clients ---


@business_mcp.tool(
    name="clients_list",
    description="List clients. Optional search by name/company. Returns id, name, company, email, phone, address, city.",
)
async def business_clients_list(
    search: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    params = _clean_params(search=search, limit=limit)
    return await call_business_api("GET", "/api/v1/clients/", params=params)


@business_mcp.tool(
    name="clients_get",
    description="Get a single client by ID with all details.",
)
async def business_clients_get(client_id: int) -> Dict[str, Any]:
    return await call_business_api("GET", f"/api/v1/clients/{client_id}")


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
) -> Dict[str, Any]:
    body = {"name": name, "country": country}
    for k, v in {"company": company, "email": email, "phone": phone, "address": address,
                  "zip": zip, "city": city, "uid_number": uid_number, "notes": notes}.items():
        if v is not None:
            body[k] = v
    return await call_business_api("POST", "/api/v1/clients/", json_body=body)


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
) -> Dict[str, Any]:
    body = {}
    for k, v in {"name": name, "company": company, "email": email, "phone": phone,
                  "address": address, "zip": zip, "city": city, "country": country,
                  "uid_number": uid_number, "notes": notes, "lead_status": lead_status,
                  "source": source, "next_followup_at": next_followup_at}.items():
        if v is not None:
            body[k] = v
    return await call_business_api("PATCH", f"/api/v1/clients/{client_id}", json_body=body)


# --- CRM ---


@business_mcp.tool(
    name="clients_update_status",
    description="Update a client's lead status. Statuses: lead, prospect, active, inactive, lost.",
)
async def business_clients_update_status(
    client_id: int,
    lead_status: str,
) -> Dict[str, Any]:
    return await call_business_api(
        "PATCH", f"/api/v1/clients/{client_id}",
        json_body={"lead_status": lead_status},
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
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"interaction_type": interaction_type, "subject": subject}
    if description:
        body["description"] = description
    return await call_business_api(
        "POST", f"/api/v1/clients/{client_id}/interactions", json_body=body,
    )


@business_mcp.tool(
    name="clients_interactions",
    description="List recent interactions for a client. Returns type, subject, date.",
)
async def business_clients_interactions(
    client_id: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    params = _clean_params(limit=limit)
    return await call_business_api(
        "GET", f"/api/v1/clients/{client_id}/interactions", params=params,
    )


@business_mcp.tool(
    name="clients_pipeline",
    description="CRM pipeline overview: clients grouped by lead status (lead, prospect, active, inactive, lost) with counts.",
)
async def business_clients_pipeline() -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/crm/pipeline")


@business_mcp.tool(
    name="clients_followups",
    description="List clients with overdue or upcoming followups (next 7 days). Returns overdue and upcoming lists.",
)
async def business_clients_followups() -> Dict[str, Any]:
    return await call_business_api("GET", "/api/v1/crm/followups")


@business_mcp.tool(
    name="clients_set_followup",
    description="Set next followup date for a client. Date format: YYYY-MM-DD.",
)
async def business_clients_set_followup(
    client_id: int,
    next_followup_at: str,
) -> Dict[str, Any]:
    return await call_business_api(
        "PATCH", f"/api/v1/clients/{client_id}",
        json_body={"next_followup_at": next_followup_at},
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
) -> List[Dict[str, Any]]:
    params = _clean_params(doc_type=doc_type, status=status, client_id=client_id, year=year, limit=limit)
    return await call_business_api("GET", "/api/v1/documents/", params=params)


@business_mcp.tool(
    name="documents_get",
    description="Get a single document with all line items and client details.",
)
async def business_documents_get(doc_id: int) -> Dict[str, Any]:
    return await call_business_api("GET", f"/api/v1/documents/{doc_id}")


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
    return await call_business_api("POST", "/api/v1/documents/honorarnote", json_body=body)


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
    return await call_business_api("POST", "/api/v1/documents/invoice", json_body=body)


@business_mcp.tool(
    name="documents_mark_paid",
    description="Mark a document as paid. Optional paid_date (YYYY-MM-DD, defaults to today). Automatically creates an income transaction.",
)
async def business_documents_mark_paid(
    doc_id: int,
    paid_date: Optional[str] = None,
) -> Dict[str, Any]:
    body = {}
    if paid_date:
        body["paid_date"] = paid_date
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/mark-paid", json_body=body)


@business_mcp.tool(
    name="documents_regenerate_pdf",
    description="Regenerate the PDF for an existing document (e.g. after template changes).",
)
async def business_documents_regenerate_pdf(doc_id: int) -> Dict[str, Any]:
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/regenerate-pdf")


@business_mcp.tool(
    name="documents_send",
    description="Send a document (Honorarnote/Invoice) to client via email with PDF attachment. Optional: recipient_email override (otherwise uses client email). Updates status to 'sent'.",
)
async def business_documents_send(
    doc_id: int,
    recipient_email: Optional[str] = None,
) -> Dict[str, Any]:
    body = {}
    if recipient_email:
        body["recipient_email"] = recipient_email
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/send", json_body=body)


@business_mcp.tool(
    name="documents_cancel",
    description="Cancel a document (status → cancelled).",
)
async def business_documents_cancel(doc_id: int) -> Dict[str, Any]:
    return await call_business_api("POST", f"/api/v1/documents/{doc_id}/cancel")


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
) -> List[Dict[str, Any]]:
    params = _clean_params(tx_type=tx_type, year=year, category=category, limit=limit)
    return await call_business_api("GET", "/api/v1/transactions", params=params)


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
) -> Dict[str, Any]:
    body = {"tx_type": tx_type, "amount": amount, "vat_rate": vat_rate}
    for k, v in {"category": category, "description": description, "tx_date": tx_date, "client_id": client_id, "receipt_url": receipt_url}.items():
        if v is not None:
            body[k] = v
    return await call_business_api("POST", "/api/v1/transactions", json_body=body)


@business_mcp.tool(
    name="transactions_delete",
    description="Delete a transaction by ID. Only works for transactions not linked to documents.",
)
async def business_transactions_delete(tx_id: int) -> Dict[str, str]:
    await call_business_api("DELETE", f"/api/v1/transactions/{tx_id}")
    return {"status": "deleted", "id": str(tx_id)}


# --- Categories ---


@business_mcp.tool(
    name="categories_list",
    description="List transaction categories. Optional filter by cat_type (income/expense).",
)
async def business_categories_list(cat_type: Optional[str] = None) -> List[Dict[str, Any]]:
    params = _clean_params(cat_type=cat_type)
    return await call_business_api("GET", "/api/v1/categories", params=params)


# --- Service Health ---


@business_mcp.tool(
    name="service_health",
    description="Health check for Business API.",
)
async def business_service_health() -> Dict[str, Any]:
    return await call_business_api("GET", "/health")


business_app = business_mcp.streamable_http_app()
app.mount(BUSINESS_PATH, business_app)


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

    Args:
        to: Recipient email address
        subject: Email subject
        body: Plain text body
        source: Source identity (default: "arkturian"). Options: "arkturian", "spreadyourwings"
        template: Optional template name (e.g. "honorarnote_send", "invoice_send", "payment_reminder", "notification")
        template_data: Optional dict of data for template rendering
    """,
)
async def comm_send_email(
    to: str,
    subject: str,
    body: str = "",
    source: str = "arkturian",
    template: Optional[str] = None,
    template_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    json_body: Dict[str, Any] = {
        "source": source,
        "to": to,
        "subject": subject,
        "body": body,
    }
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
    """,
)
async def comm_send_telegram(
    message: str,
    chat_id: Optional[str] = None,
    to: Optional[str] = None,
) -> Dict[str, Any]:
    json_body: Dict[str, Any] = {"message": message}
    if chat_id:
        json_body["chat_id"] = chat_id
    if to:
        json_body["to"] = to
    return await call_comm_api("POST", "/api/v1/telegram/send", json_body=json_body)


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
async def comm_notify_human(message: str) -> Dict[str, Any]:
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "sent": False}
    try:
        result = await call_comm_api(
            "POST",
            "/api/v1/telegram/interventions/notification",
            json_body={"message": message},
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
) -> Dict[str, Any]:
    if not COMM_API_KEY:
        return {"error": "COMM_API_KEY not configured", "response": None}
    try:
        if options and len(options) > 0:
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/approval",
                json_body={
                    "message": question,
                    "options": options,
                    "timeout_seconds": timeout_seconds,
                },
            )
        else:
            create_result = await call_comm_api(
                "POST",
                "/api/v1/telegram/interventions/text-input",
                json_body={
                    "message": question,
                    "timeout_seconds": timeout_seconds,
                },
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


comm_app = comm_mcp.streamable_http_app()
app.mount(COMM_PATH, comm_app)


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


@app.on_event("startup")
async def startup() -> None:
    await _storage_stack.enter_async_context(storage_mcp.session_manager.run())
    await storage_app.router.startup()

    await _oneal_stack.enter_async_context(oneal_mcp.session_manager.run())
    await oneal_app.router.startup()

    await _oneal_storage_stack.enter_async_context(oneal_storage_mcp.session_manager.run())
    await oneal_storage_app.router.startup()

    await _artrack_stack.enter_async_context(artrack_mcp.session_manager.run())
    await artrack_app.router.startup()

    await _codepilot_stack.enter_async_context(codepilot_mcp.session_manager.run())
    await codepilot_app.router.startup()

    await _content_stack.enter_async_context(content_mcp.session_manager.run())
    await content_app.router.startup()

    await _tree_stack.enter_async_context(tree_mcp.session_manager.run())
    await tree_app.router.startup()

    await _ai_stack.enter_async_context(ai_mcp.session_manager.run())
    await ai_app.router.startup()

    await _tarot_stack.enter_async_context(tarot_mcp.session_manager.run())
    await tarot_app.router.startup()

    await _business_stack.enter_async_context(business_mcp.session_manager.run())
    await business_app.router.startup()

    await _comm_stack.enter_async_context(comm_mcp.session_manager.run())
    await comm_app.router.startup()

    await _knowledge_stack.enter_async_context(knowledge_mcp.session_manager.run())
    await knowledge_app.router.startup()


@app.on_event("shutdown")
async def shutdown() -> None:
    await knowledge_app.router.shutdown()
    await _knowledge_stack.aclose()

    await comm_app.router.shutdown()
    await _comm_stack.aclose()

    await business_app.router.shutdown()
    await _business_stack.aclose()

    await tarot_app.router.shutdown()
    await _tarot_stack.aclose()

    await ai_app.router.shutdown()
    await _ai_stack.aclose()

    await tree_app.router.shutdown()
    await _tree_stack.aclose()

    await content_app.router.shutdown()
    await _content_stack.aclose()

    await codepilot_app.router.shutdown()
    await _codepilot_stack.aclose()

    await artrack_app.router.shutdown()
    await _artrack_stack.aclose()

    await oneal_storage_app.router.shutdown()
    await _oneal_storage_stack.aclose()

    await oneal_app.router.shutdown()
    await _oneal_stack.aclose()

    await storage_app.router.shutdown()
    await _storage_stack.aclose()


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
