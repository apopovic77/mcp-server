#!/usr/bin/env python3
"""
Arkturian MCP Server

Exposes three MCP endpoint groups over HTTP/SSE with per-tenant isolation:
  • /storage        – Arkturian tenant Storage & Knowledge Graph tools
  • /oneal          – O'Neal product catalogue API
  • /oneal-storage  – O'Neal tenant Storage & Knowledge Graph tools (615 products)
"""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

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
    url = httpx.URL(f"{STORAGE_API_BASE}/storage/media/{id}").copy_add_params(options)
    return {"url": str(url), "parameters": options}


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
    - storage.media_url: Transformable media URL (https://api.arkturian.com/storage/media/{id})
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
    url = httpx.URL(f"{STORAGE_API_BASE}/storage/media/{id}").copy_add_params(options)
    return {"url": str(url), "parameters": options}


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


# --------------------------------------------------------------------------- #
# FastAPI wrapper
# --------------------------------------------------------------------------- #

app = FastAPI(title="arkturian-mcp", version="2.2.0", description="Arkturian MCP Aggregator with per-tenant isolation")

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
app.mount(STORAGE_PATH, storage_app)
app.mount(ONEAL_PATH, oneal_app)
app.mount(ONEAL_STORAGE_PATH, oneal_storage_app)

_storage_stack = AsyncExitStack()
_oneal_stack = AsyncExitStack()
_oneal_storage_stack = AsyncExitStack()


@app.on_event("startup")
async def startup() -> None:
    await _storage_stack.enter_async_context(storage_mcp.session_manager.run())
    await storage_app.router.startup()

    await _oneal_stack.enter_async_context(oneal_mcp.session_manager.run())
    await oneal_app.router.startup()

    await _oneal_storage_stack.enter_async_context(oneal_storage_mcp.session_manager.run())
    await oneal_storage_app.router.startup()


@app.on_event("shutdown")
async def shutdown() -> None:
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
        "version": "2.2.0",
        "description": "Arkturian MCP Aggregator with per-tenant isolation",
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
                    "version": "2.2.0",
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
