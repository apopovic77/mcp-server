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

# Telegram API for human-in-the-loop
TELEGRAM_API_BASE = os.getenv("TELEGRAM_API_BASE", "https://telegram-api.arkturian.com")
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY", "")

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
) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params or {})
            elif method == "POST":
                response = await client.post(url, headers=headers, params=params or {}, json=json_body or {})
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


async def call_telegram_api(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> Any:
    """Call Telegram intervention API for human-in-the-loop."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            url = f"{TELEGRAM_API_BASE}{endpoint}"
            headers = {"X-API-Key": TELEGRAM_API_KEY}

            if method == "GET":
                response = await client.get(url, headers=headers, params=params or {})
            elif method == "POST":
                response = await client.post(url, headers=headers, json=json_body or {})
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Telegram API error %s %s: %s", method, url, exc.response.text)
            raise
        except httpx.HTTPError as exc:
            logger.error("Telegram API request to %s failed: %s", url, exc)
            raise


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
    """Send notification to human via Telegram."""
    if not TELEGRAM_API_KEY:
        return {"error": "TELEGRAM_API_KEY not configured", "sent": False}

    try:
        result = await call_telegram_api(
            "POST",
            "/interventions/notification",
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
    """Ask human a question and wait for response."""
    if not TELEGRAM_API_KEY:
        return {"error": "TELEGRAM_API_KEY not configured", "response": None}

    try:
        # Determine endpoint based on options
        if options and len(options) > 0:
            # Approval request with buttons
            create_result = await call_telegram_api(
                "POST",
                "/interventions/approval",
                json_body={
                    "message": question,
                    "options": options,
                    "timeout_seconds": timeout_seconds,
                },
            )
        else:
            # Text input request
            create_result = await call_telegram_api(
                "POST",
                "/interventions/text-input",
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

            wait_result = await call_telegram_api(
                "GET",
                f"/interventions/{request_id}/wait",
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
    name="service_health",
    description="Health check for the Content API.",
)
async def content_service_health() -> Dict[str, Any]:
    return await call_content_api("GET", "/health")


# --------------------------------------------------------------------------- #
# FastAPI wrapper
# --------------------------------------------------------------------------- #

app = FastAPI(title="arkturian-mcp", version="2.5.0", description="Arkturian MCP Aggregator with AI generation and Content API")

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
app.mount(STORAGE_PATH, storage_app)
app.mount(ONEAL_PATH, oneal_app)
app.mount(ONEAL_STORAGE_PATH, oneal_storage_app)
app.mount(ARTRACK_PATH, artrack_app)
app.mount(CODEPILOT_PATH, codepilot_app)
app.mount(CONTENT_PATH, content_app)
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

_storage_stack = AsyncExitStack()
_oneal_stack = AsyncExitStack()
_oneal_storage_stack = AsyncExitStack()
_artrack_stack = AsyncExitStack()
_codepilot_stack = AsyncExitStack()
_content_stack = AsyncExitStack()
_ai_stack = AsyncExitStack()


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

    await _ai_stack.enter_async_context(ai_mcp.session_manager.run())
    await ai_app.router.startup()


@app.on_event("shutdown")
async def shutdown() -> None:
    await ai_app.router.shutdown()
    await _ai_stack.aclose()

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
        "version": "2.5.0",
        "description": "Arkturian MCP Aggregator with per-tenant isolation, human-in-the-loop, AI generation, and Content API",
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
                "upstream": TELEGRAM_API_BASE,
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
                    "version": "2.5.0",
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
