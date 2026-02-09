# cloudrun/main.py

from fastapi import FastAPI, HTTPException, Response, Request, WebSocket, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
from google.cloud import bigquery
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import time
from zoneinfo import ZoneInfo
import logging
import requests
from cachetools import TTLCache
from cachetools.keys import hashkey
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from services.input_validator import InputValidator
from services.output_validator import OutputValidator
from services.rate_limiter import BudgetGuard, limiter
from services.feed_definitions import FeedDefinitions
from services.security_monitor import SecurityMonitor
from services.content_assistant import ContentAssistant
from services.vacation_limitter import VacationLimiter
from services.briefing_service import BriefingService
from services.secret_manager import get_secret_value
from services.lecture_service import LectureService
from config import Config

# OpenTelemetry Imports (DISABLED - Caused Startup Failure)
# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelmetry.sdk.trace.export import BatchSpanProcessor
# from opentelmetry.exporter.cloud_trace import CloudTraceSpanExporter
# from opentelmetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize Tracing (DISABLED)
# Use Cloud Trace Exporter for production (Cloud Run)
# try:
#     tracer_provider = TracerProvider()
#     cloud_trace_exporter = CloudTraceSpanExporter()
#     tracer_provider.add_span_processor(
#         BatchSpanProcessor(cloud_trace_exporter)
#     )
#     trace.set_tracer_provider(tracer_provider)
#     tracer = trace.get_tracer(__name__)
#     logger.info("OpenTelemetry configured with Cloud Trace Exporter.")
# except Exception as e:
#     logger.error(f"Failed to configure OpenTelemetry: {e}")
#     # Fallback/No-op if exporter fails (e.g. local without credentials)
#     from opentelemetry.sdk.trace import TracerProvider
#     trace.set_tracer_provider(TracerProvider())
#     tracer = trace.get_tracer(__name__)

# MOCK TRACER (Restored for Stability)
from contextlib import contextmanager

class MockSpan:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def set_attribute(self, key, value): pass

class MockTracer:
    def start_as_current_span(self, name):
        return MockSpan()

tracer = MockTracer()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Briefing start idempotency cache (per client, short window)
briefing_start_cache = TTLCache(maxsize=1024, ttl=600)

# Initialize FastAPI app
app = FastAPI(
    title="Content Intelligence Platform API",
    description="AI-powered RSS feed curation with 106 curated feeds",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# CORS (Allow devcaddie.com and localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://devcaddie.com", 
        "https://www.devcaddie.com"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Instrument FastAPI
# FastAPIInstrumentor.instrument_app(app)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    bot_start_origin = os.getenv("BOT_START_ORIGIN")
    bot_connect = f" {bot_start_origin}" if bot_start_origin else ""
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://c.daily.co blob:; "
        "script-src-elem 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://c.daily.co blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "media-src 'self' blob:; "
        "worker-src 'self' blob:; "
        f"connect-src 'self' https://api.github.com https://*.daily.co wss://*.daily.co https://o77906.ingest.sentry.io https://cdn.jsdelivr.net{bot_connect}; "
        "frame-ancestors 'self' https://dev.to;"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "ALLOW-FROM https://dev.to" # For older browsers
    return response

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Initialize BigQuery client
PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    try:
        import google.auth
        _, PROJECT_ID = google.auth.default()
    except Exception:
        PROJECT_ID = None
    if not PROJECT_ID:
        logger.warning("PROJECT_ID not set, defaulting to internal lookup failed")
bq_client = bigquery.Client(project=PROJECT_ID)
DATASET_ID = "content_intelligence"
LLM_DATASET_ID = "llm_observability"
BRIEFING_METRICS_DATASET_ID = os.getenv("BRIEFING_METRICS_BQ_DATASET", "llm_observability")
BRIEFING_METRICS_TABLE_ID = os.getenv("BRIEFING_METRICS_BQ_TABLE", "briefing_metrics_raw")
LLM_TABLE_ID = "metrics"


# Pydantic Models
class Feed(BaseModel):
    """RSS Feed model."""
    title: str
    xml_url: HttpUrl
    html_url: Optional[HttpUrl] = None
    description: Optional[str] = ""


class FeedWithMetadata(Feed):
    """Feed with additional metadata."""
    feed_id: str
    category: str
    is_active: bool
    last_fetched_at: Optional[datetime] = None
    fetch_error_count: int = 0


class SourcesResponse(BaseModel):
    """Response model for /api/sources endpoint."""
    total_feeds: int
    categories: List[str]
    by_category: Dict[str, List[Feed]]


class StatsResponse(BaseModel):
    """Response model for /api/sources/stats endpoint."""
    total_feeds: int
    total_categories: int
    feeds_fetched_24h: int
    total_errors: int
    estimated_articles_per_day: int

# Initialize Security Monitor
security_monitor = SecurityMonitor(project_id=PROJECT_ID)

# Initialize Services
content_assistant = ContentAssistant(project_id=PROJECT_ID, location="us-central1")
briefing_service = BriefingService(project_id=PROJECT_ID, location="us-central1")
lecture_service = LectureService(project_id=PROJECT_ID, location="us-central1")


# Routes

@app.get("/")
async def root():
    """Root endpoint - serve the single-page application."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/sources", response_class=HTMLResponse)
async def sources_page():
    """Serve the sources HTML page."""
    try:
        with open("static/sources.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sources page not found")


@app.get("/api/sources", response_model=SourcesResponse)
async def get_sources():
    """
    Get all active feeds grouped by category.
    
    Returns:
        JSON with feeds organized by category
    """
    try:
        query = f"""
        SELECT 
            title,
            xml_url,
            html_url,
            category,
            description
        FROM `{PROJECT_ID}.{DATASET_ID}.feeds_metadata`
        WHERE is_active = TRUE
        ORDER BY category, title
        """
        
        feeds_by_category = {}
        
        for row in bq_client.query(query).result():
            category = row.category
            
            if category not in feeds_by_category:
                feeds_by_category[category] = []
            
            feeds_by_category[category].append({
                "title": row.title,
                "xml_url": row.xml_url,
                "html_url": row.html_url,
                "description": row.description or ""
            })
        
        total_feeds = sum(len(feeds) for feeds in feeds_by_category.values())
        
        return {
            "total_feeds": total_feeds,
            "categories": list(feeds_by_category.keys()),
            "by_category": feeds_by_category
        }
    
    except Exception as e:
        logger.error(f"Error fetching sources: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching sources: {str(e)}")


@app.get("/api/sources/category/{category}")
async def get_feeds_by_category(category: str):
    """
    Get feeds for a specific category.
    
    Args:
        category: Category name (e.g., "Data Engineering")
    """
    try:
        query = f"""
        SELECT 
            title,
            xml_url,
            html_url,
            description
        FROM `{PROJECT_ID}.{DATASET_ID}.feeds_metadata`
        WHERE is_active = TRUE 
            AND category = @category
        ORDER BY title
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("category", "STRING", category)
            ]
        )
        
        feeds = []
        for row in bq_client.query(query, job_config=job_config).result():
            feeds.append({
                "title": row.title,
                "xml_url": row.xml_url,
                "html_url": row.html_url,
                "description": row.description or ""
            })
        
        if not feeds:
            raise HTTPException(status_code=404, detail=f"Category '{category}' not found or has no feeds")
        
        return {
            "category": category,
            "feed_count": len(feeds),
            "feeds": feeds
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching category feeds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources/opml")
async def download_opml():
    """
    Generate and serve OPML file from BigQuery feeds.
    
    Returns:
        OPML file for download
    """
    try:
        query = f"""
        SELECT 
            title,
            xml_url,
            html_url,
            category,
            description
        FROM `{PROJECT_ID}.{DATASET_ID}.feeds_metadata`
        WHERE is_active = TRUE
        ORDER BY category, title
        """
        
        # Create OPML structure
        opml = ET.Element('opml', version='2.0')
        head = ET.SubElement(opml, 'head')
        title = ET.SubElement(head, 'title')
        title.text = 'Content Intelligence Platform Feeds'
        
        date_created = ET.SubElement(head, 'dateCreated')
        date_created.text = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
        
        body = ET.SubElement(opml, 'body')
        
        # Group by category
        feeds_by_category = {}
        for row in bq_client.query(query).result():
            category = row.category
            if category not in feeds_by_category:
                feeds_by_category[category] = []
            feeds_by_category[category].append(row)
        
        # Add categories and feeds
        for category, feeds in sorted(feeds_by_category.items()):
            category_outline = ET.SubElement(body, 'outline', text=category, title=category)
            
            for feed in feeds:
                ET.SubElement(
                    category_outline,
                    'outline',
                    type='rss',
                    text=feed.title,
                    title=feed.title,
                    xmlUrl=feed.xml_url,
                    htmlUrl=feed.html_url or '',
                    description=feed.description or ''
                )
        
        # Convert to XML string
        xml_str = ET.tostring(opml, encoding='utf-8', method='xml')
        xml_declaration = b'<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content = xml_declaration + xml_str
        
        return Response(
            content=xml_content,
            media_type='application/xml',
            headers={
                'Content-Disposition': 'attachment; filename="content-intelligence-feeds.opml"'
            }
        )
    
    except Exception as e:
        logger.error(f"Error generating OPML: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating OPML: {str(e)}")


@app.get("/api/sources/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get feed statistics.
    
    Returns:
        Statistics about feeds, fetches, and errors
    """
    try:
        stats_query = f"""
        SELECT
            COUNT(*) as total_feeds,
            COUNT(DISTINCT category) as total_categories,
            SUM(CASE WHEN last_fetched_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR) 
                THEN 1 ELSE 0 END) as feeds_fetched_24h,
            SUM(fetch_error_count) as total_errors
        FROM `{PROJECT_ID}.{DATASET_ID}.feeds_metadata`
        WHERE is_active = TRUE
        """
        
        result = list(bq_client.query(stats_query).result())[0]
        
        return {
            "total_feeds": result.total_feeds,
            "total_categories": result.total_categories,
            "feeds_fetched_24h": result.feeds_fetched_24h,
            "total_errors": result.total_errors,
            "estimated_articles_per_day": result.total_feeds * 10  # Rough estimate
        }
    
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/content-stats")
async def get_content_stats(days: int = 7):
    """
    Get dynamic content filtering statistics (Before vs After AI Filter).
    """
    try:
        query = f"""
        SELECT
            COUNT(*) as total_processed,
            SUM(CASE WHEN final_score >= 80 THEN 1 ELSE 0 END) as highly_relevant
        FROM `{PROJECT_ID}.{DATASET_ID}.articles_scored`
        WHERE scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )
        
        row = list(bq_client.query(query, job_config=job_config).result())[0]
        
        avg_time_per_article = 5  # minutes
        summary_time = 1 # minutes
        
        time_saved_minutes = (row.total_processed * avg_time_per_article) - (row.highly_relevant * summary_time)
        hours_saved = round(time_saved_minutes / 60)
        
        return {
            "period_days": days,
            "total_processed": row.total_processed,
            "highly_relevant": row.highly_relevant,
            "filtering_ratio": f"{round((row.highly_relevant / row.total_processed * 100), 1)}%" if row.total_processed > 0 else "0%",
            "hours_saved_est": hours_saved
        }

    except Exception as e:
        logger.error(f"Error fetching content stats: {e}")
        # Return fallback data if query fails
        return {
            "total_processed": 500 * days,
            "highly_relevant": 20 * days,
            "filtering_ratio": "4.0%",
            "hours_saved_est": 2 * days
        }


@app.get("/api/sources/categories")
async def get_categories():
    """
    Get list of all categories with feed counts.

    Returns:
        List of categories with metadata
    """
    try:
        query = f"""
        SELECT
            category,
            COUNT(*) as feed_count
        FROM `{PROJECT_ID}.{DATASET_ID}.feeds_metadata`
        WHERE is_active = TRUE
        GROUP BY category
        ORDER BY category
        """

        categories = []
        for row in bq_client.query(query).result():
            categories.append({
                "name": row.category,
                "feed_count": row.feed_count
            })

        return {
            "total_categories": len(categories),
            "categories": categories
        }

    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ARTICLE ENDPOINTS - Curated content with dual scoring (AI + Community)
# ============================================================================

@app.get("/api/articles/for-you")
async def get_articles_for_you(limit: int = 10, min_score: int = 80):
    """
    Get today's curated articles (high AI relevance + community validation).

    Default view showing top articles matching your interests.

    Args:
        limit: Maximum articles to return (default 20)
        min_score: Minimum final score (default 80 for must-reads)
    """
    try:
        query = f"""
        SELECT
            article_id,
            title,
            url,
            summary,
            ai_relevance_score,
            community_score,
            final_score,
            score_category,
            hn_points,
            hn_comments,
            hn_url,
            is_trending,
            is_viral,
            key_topics,
            ai_reasoning,
            category,
            scored_at
        FROM `{PROJECT_ID}.{DATASET_ID}.articles_scored`
        WHERE scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          AND final_score >= @min_score
        QUALIFY ROW_NUMBER() OVER (PARTITION BY article_id ORDER BY final_score DESC, scored_at DESC) = 1
        ORDER BY final_score DESC, scored_at DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("min_score", "INT64", min_score),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        articles = []
        for row in bq_client.query(query, job_config=job_config).result():
            articles.append({
                "article_id": row.article_id,
                "title": row.title,
                "url": row.url,
                "summary": row.summary,
                "scores": {
                    "ai_relevance": row.ai_relevance_score,
                    "community": row.community_score,
                    "final": row.final_score,
                    "category": row.score_category
                },
                "social_proof": {
                    "hn_points": row.hn_points,
                    "hn_comments": row.hn_comments,
                    "hn_url": row.hn_url,
                    "is_trending": row.is_trending,
                    "is_viral": row.is_viral
                },
                "topics": row.key_topics,
                "ai_reasoning": row.ai_reasoning,
                "category": row.category,
                "scored_at": row.scored_at.isoformat() if row.scored_at else None
            })

        return {
            "view": "for-you",
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles/trending")
async def get_trending_articles(limit: int = 10):
    """
    Get trending/viral articles regardless of personal relevance.

    Shows what's hot in tech right now - articles with high community
    engagement (HN upvotes, comments) even if not matching your profile.

    Args:
        limit: Maximum articles to return (default 10)
    """
    try:
        query = f"""
        SELECT
            article_id,
            title,
            url,
            summary,
            ai_relevance_score,
            community_score,
            final_score,
            hn_points,
            hn_comments,
            hn_url,
            lobsters_points,
            lobsters_comments,
            lobsters_url,
            is_viral,
            category,
            scored_at,
            ai_reasoning,
            actionability,
            estimated_read_time,
            content_type
        FROM `{PROJECT_ID}.{DATASET_ID}.articles_scored`
        WHERE is_trending = TRUE
          AND ai_relevance_score >= 25  -- Junk floor: no low-quality viral content
          AND scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY url ORDER BY community_score DESC, scored_at DESC) = 1
        ORDER BY community_score DESC, hn_points DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        articles = []
        for row in bq_client.query(query, job_config=job_config).result():
            articles.append({
                "article_id": row.article_id,
                "title": row.title,
                "url": row.url,
                "summary": row.summary,
                "scores": {
                    "ai_relevance": row.ai_relevance_score,
                    "community": row.community_score,
                    "final": row.final_score
                },
                "social_proof": {
                    "hn_points": row.hn_points,
                    "hn_comments": row.hn_comments,
                    "hn_url": row.hn_url,
                    "lobsters_points": row.lobsters_points,
                    "lobsters_comments": row.lobsters_comments,
                    "lobsters_url": row.lobsters_url,
                    "is_viral": row.is_viral
                },
                "category": row.category,
                "ai_reasoning": row.ai_reasoning,
                "actionability": row.actionability,
                "estimated_read_time": row.estimated_read_time,
                "content_type": row.content_type,
                "scored_at": row.scored_at.isoformat() if row.scored_at else None
            })

        return {
            "view": "trending",
            "description": "Hot articles in tech this week (high community engagement)",
            "count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Error fetching trending articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles/recent")
async def get_recent_articles(
    days: int = 7, 
    min_score: int = 60, 
    category: Optional[str] = None,
    limit: int = 200
):
    """
    Get scored articles from the last N days (Default: 7).
    Useful for 'Daily Reads' (days=1) or 'Weekly Overview' (days=7).
    """
    try:
        query = f"""
        SELECT
            article_id,
            title,
            url,
            summary,
            ai_relevance_score,
            community_score,
            final_score,
            score_category,
            hn_points,
            is_trending,
            key_topics,
            category,
            scored_at,
            ai_reasoning,
            actionability,
            estimated_read_time,
            content_type
        FROM `{PROJECT_ID}.{DATASET_ID}.articles_scored`
        WHERE scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
          AND final_score >= @min_score
          AND (@category IS NULL OR category = @category)
        ORDER BY scored_at DESC, final_score DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("days", "INT64", days),
                bigquery.ScalarQueryParameter("min_score", "INT64", min_score),
                bigquery.ScalarQueryParameter("category", "STRING", category),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        articles = []
        for row in bq_client.query(query, job_config=job_config).result():
            articles.append({
                "article_id": row.article_id,
                "title": row.title,
                "url": row.url,
                "summary": row.summary,
                "scores": {
                    "ai_relevance": row.ai_relevance_score,
                    "community": row.community_score,
                    "final": row.final_score
                },
                "final_score": row.final_score,
                "social_proof": {
                    "hn_points": row.hn_points,
                    "is_trending": row.is_trending
                },
                "score_category": row.score_category,
                "topics": row.key_topics,
                "category": row.category or "Uncategorized",
                "ai_reasoning": row.ai_reasoning,
                "actionability": row.actionability,
                "estimated_read_time": row.estimated_read_time,
                "content_type": row.content_type,
                "scored_at": row.scored_at.isoformat() if row.scored_at else None
            })

        return {
            "view": "recent",
            "days": days,
            "filter_category": category,
            "count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Error fetching recent articles: {e}")
        # Return empty list on error (e.g. table missing)
        return {"articles": [], "error": str(e)}


# ============================================================================
# LLM OBSERVABILITY - Direct BigQuery Access with Caching
# ============================================================================

# Cache results for 5 minutes (300 seconds)
# Using a simple in-memory cache since Cloud Run instances are ephemeral anyway
dashboard_cache = TTLCache(maxsize=100, ttl=300)

async def get_cached_kpis(days: int):
    """Helper to get KPIs with caching manually managed"""
    key = hashkey('kpis', days)
    if key in dashboard_cache:
        return dashboard_cache[key]
    
    # Cache miss - fetch
    result = await fetch_kpis_from_bq(days)
    dashboard_cache[key] = result
    return result

async def fetch_kpis_from_bq(days: int):
    """Actual BigQuery fetch for KPIs"""
    table_ref = f"{PROJECT_ID}.{LLM_DATASET_ID}.{LLM_TABLE_ID}"
    
    query = f"""
    WITH stats AS (
        SELECT 
            metric_name,
            value,
            timestamp
        FROM `{table_ref}`
        WHERE timestamp > UNIX_SECONDS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY))
    )
    SELECT
        COUNTIF(metric_name = 'llm.reviews.total') as total_reviews,
        COALESCE(SUM(CASE WHEN metric_name = 'llm.cost.per_review' THEN value ELSE 0 END), 0) as total_cost,
        COALESCE(AVG(CASE WHEN metric_name = 'llm.duration.seconds' THEN value END), 0) as avg_latency,
        COALESCE(AVG(CASE WHEN metric_name = 'rag.retrieval.max_similarity' THEN value END), 0) as avg_rag_similarity,
        COUNTIF(metric_name = 'llm.errors') as total_errors
    FROM stats
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", days),
        ]
    )
    
    # Run in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(
        None, 
        lambda: list(bq_client.query(query, job_config=job_config).result())[0]
    )
    
    # Calculate error rate
    reviews = row.total_reviews or 1
    error_rate = (row.total_errors / reviews) * 100
    
    return {
        "total_reviews": row.total_reviews,
        "total_cost": round(row.total_cost, 4),
        "avg_latency": round(row.avg_latency, 2),
        "rag_similarity": round(row.avg_rag_similarity, 3),
        "error_rate": round(error_rate, 2)
    }

async def get_cached_timeseries(days: int):
    """Helper to get Timeseries with caching"""
    key = hashkey('timeseries', days)
    if key in dashboard_cache:
        return dashboard_cache[key]
        
    result = await fetch_timeseries_from_bq(days)
    dashboard_cache[key] = result
    return result

async def fetch_timeseries_from_bq(days: int):
    table_ref = f"{PROJECT_ID}.{LLM_DATASET_ID}.{LLM_TABLE_ID}"
    
    query = f"""
    SELECT 
        DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) as date,
        COUNTIF(metric_name = 'llm.reviews.total') as reviews,
        COALESCE(SUM(CASE WHEN metric_name = 'llm.cost.per_review' THEN value ELSE 0 END), 0) as cost,
        COALESCE(AVG(CASE WHEN metric_name = 'llm.duration.seconds' THEN value END), 0) as avg_latency,
        COALESCE(AVG(CASE WHEN metric_name = 'rag.retrieval.max_similarity' THEN value END), 0) as rag_similarity
    FROM `{table_ref}`
    WHERE timestamp > UNIX_SECONDS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY))
    GROUP BY 1
    ORDER BY 1 ASC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", days),
        ]
    )
    
    loop = asyncio.get_event_loop()
    job_result = await loop.run_in_executor(
        None,
        lambda: list(bq_client.query(query, job_config=job_config).result())
    )
    
    results = []
    for row in job_result:
        results.append({
            "date": row.date.isoformat(),
            "reviews": row.reviews,
            "cost": round(row.cost, 4),
            "avg_latency": round(row.avg_latency, 2),
            "rag_similarity": round(row.rag_similarity, 3)
        })
    return results

@app.get("/api/llm-observability/kpis")
async def get_llm_kpis(days: int = 7):
    """Get high-level KPIs directly from BigQuery (Cached)"""
    try:
        return await get_cached_kpis(days)
    except Exception as e:
        logger.error(f"Error fetching LLM KPIs: {e}")
        return {
            "total_reviews": 0, "total_cost": 0, 
            "avg_latency": 0, "rag_similarity": 0, "error_rate": 0
        }

@app.get("/api/llm-observability/timeseries")
async def get_llm_timeseries(days: int = 30):
    """Get timeseries data directly from BigQuery (Cached)"""
    try:
        return await get_cached_timeseries(days)
    except Exception as e:
        logger.error(f"Error fetching LLM Timeseries: {e}")
        return []


# ============================================================================
# BRIEFING OBSERVABILITY - BigQuery-backed
# ============================================================================

async def fetch_briefing_kpis_from_bq(days: int) -> dict:
    table_ref = f"{PROJECT_ID}.{BRIEFING_METRICS_DATASET_ID}.{BRIEFING_METRICS_TABLE_ID}"
    query = f"""
    SELECT
        APPROX_QUANTILES(ttfb_ms, 100)[OFFSET(50)] AS ttfb_p50_ms,
        APPROX_QUANTILES(ttfb_ms, 100)[OFFSET(95)] AS ttfb_p95_ms,
        AVG(processing_ms) AS processing_avg_ms,
        APPROX_QUANTILES(turn_e2e_ms, 100)[OFFSET(50)] AS turn_e2e_p50_ms,
        APPROX_QUANTILES(turn_e2e_ms, 100)[OFFSET(95)] AS turn_e2e_p95_ms,
        SUM(llm_total_tokens) AS llm_total_tokens,
        SUM(tts_chars) AS tts_chars,
        COUNTIF(llm_total_tokens IS NOT NULL) AS turns,
        SAFE_DIVIDE(COUNTIF(interruption), COUNTIF(llm_total_tokens IS NOT NULL)) AS interruption_rate,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(ts), SECOND) AS last_sample_age_s
    FROM `{table_ref}`
    WHERE ts > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", days)]
    )
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: list(bq_client.query(query, job_config=job_config).result())
    )
    if not rows:
        return {"ok": False, "reason": "no-metrics"}
    row = rows[0]
    if row.last_sample_age_s is None:
        return {"ok": False, "reason": "no-metrics"}
    return {
        "ok": True,
        "ttfb_p50_ms": row.ttfb_p50_ms,
        "ttfb_p95_ms": row.ttfb_p95_ms,
        "processing_avg_ms": row.processing_avg_ms,
        "turn_e2e_p50_ms": row.turn_e2e_p50_ms,
        "turn_e2e_p95_ms": row.turn_e2e_p95_ms,
        "llm_total_tokens": row.llm_total_tokens,
        "tts_chars": row.tts_chars,
        "turns": row.turns,
        "interruption_rate": row.interruption_rate,
        "last_sample_age_s": row.last_sample_age_s,
    }


async def fetch_briefing_timeseries_from_bq(days: int, limit: int) -> dict:
    table_ref = f"{PROJECT_ID}.{BRIEFING_METRICS_DATASET_ID}.{BRIEFING_METRICS_TABLE_ID}"
    # Aggregate per session (room_url) to avoid burst patterns from per-turn logging
    query = f"""
    WITH session_agg AS (
        SELECT
            room_url,
            MIN(ts) AS session_start,
            AVG(ttfb_ms) AS ttfb_ms,
            AVG(turn_e2e_ms) AS turn_e2e_ms,
            SUM(llm_total_tokens) AS llm_total_tokens
        FROM `{table_ref}`
        WHERE ts > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
          AND room_url IS NOT NULL
        GROUP BY room_url
    ),
    ranked AS (
        SELECT session_start AS ts, ttfb_ms, turn_e2e_ms, llm_total_tokens
        FROM session_agg
        ORDER BY session_start DESC
        LIMIT @limit
    )
    SELECT ts, ttfb_ms, turn_e2e_ms, llm_total_tokens
    FROM ranked
    ORDER BY ts ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", days),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: list(bq_client.query(query, job_config=job_config).result())
    )
    series = []
    for row in rows:
        ts_seconds = int(row.ts.timestamp()) if row.ts else None
        series.append({
            "ts": ts_seconds,
            "ttfb_ms": row.ttfb_ms,
            "turn_e2e_ms": row.turn_e2e_ms,
            "llm_total_tokens": row.llm_total_tokens,
        })
    return {"ok": True, "series": series}


# ============================================================================
# SCORING OBSERVABILITY - Airflow DAG metrics
# ============================================================================

SCORING_BATCH_TABLE = os.getenv("AIRFLOW_OBS_BATCH_TABLE", "scoring_batch_metrics")
SCORING_TRACE_TABLE = os.getenv("AIRFLOW_OBS_TRACE_TABLE", "scoring_trace_raw")


async def fetch_scoring_kpis_from_bq(days: int) -> dict:
    """Fetch scoring KPIs from BigQuery."""
    table_ref = f"{PROJECT_ID}.{LLM_DATASET_ID}.{SCORING_BATCH_TABLE}"
    query = f"""
    SELECT
        COUNT(*) AS total_runs,
        SUM(fetched_count) AS total_fetched,
        SUM(new_count) AS total_new,
        SUM(scored_count) AS total_scored,
        SUM(stored_count) AS total_stored,
        AVG(low_score_rate) AS avg_low_score_rate,
        AVG(final_score_mean) AS avg_final_score,
        AVG(final_score_p50) AS avg_p50_score,
        AVG(community_coverage) AS avg_community_coverage,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(ts), SECOND) AS last_run_age_s
    FROM `{table_ref}`
    WHERE ts > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", days)]
    )
    rows = await asyncio.to_thread(
        lambda: list(bq_client.query(query, job_config=job_config).result())
    )
    if not rows:
        return {"ok": False, "reason": "no-data"}
    row = rows[0]
    if row.last_run_age_s is None:
        return {"ok": False, "reason": "no-metrics"}
    return {
        "ok": True,
        "total_runs": row.total_runs,
        "total_fetched": row.total_fetched,
        "total_new": row.total_new,
        "total_scored": row.total_scored,
        "total_stored": row.total_stored,
        "avg_low_score_rate": row.avg_low_score_rate,
        "avg_final_score": row.avg_final_score,
        "avg_p50_score": row.avg_p50_score,
        "avg_community_coverage": row.avg_community_coverage,
        "last_run_age_s": row.last_run_age_s,
    }


async def fetch_scoring_timeseries_from_bq(days: int, limit: int) -> dict:
    """Fetch scoring timeseries from BigQuery."""
    table_ref = f"{PROJECT_ID}.{LLM_DATASET_ID}.{SCORING_BATCH_TABLE}"
    query = f"""
    WITH filtered AS (
        SELECT ts, dag_run_id, new_count, scored_count, stored_count,
               final_score_mean, final_score_p50, low_score_rate, community_coverage
        FROM `{table_ref}`
        WHERE ts > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        ORDER BY ts DESC
        LIMIT @limit
    )
    SELECT * FROM filtered ORDER BY ts ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", days),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    rows = await asyncio.to_thread(
        lambda: list(bq_client.query(query, job_config=job_config).result())
    )
    series = []
    for row in rows:
        ts_seconds = int(row.ts.timestamp()) if row.ts else None
        series.append({
            "ts": ts_seconds,
            "dag_run_id": row.dag_run_id,
            "new_count": row.new_count,
            "scored_count": row.scored_count,
            "stored_count": row.stored_count,
            "final_score_mean": row.final_score_mean,
            "final_score_p50": row.final_score_p50,
            "low_score_rate": row.low_score_rate,
            "community_coverage": row.community_coverage,
        })
    return {"ok": True, "series": series}


@app.get("/api/observability/scoring/kpis")
async def get_scoring_kpis(days: int = 7):
    """Get scoring observability KPIs."""
    try:
        return await fetch_scoring_kpis_from_bq(days)
    except Exception as e:
        logger.error(f"Failed to fetch scoring KPIs: {e}")
        return {"ok": False, "reason": str(e)}


@app.get("/api/observability/scoring/timeseries")
async def get_scoring_timeseries(days: int = 7, limit: int = 50):
    """Get scoring observability timeseries."""
    try:
        return await fetch_scoring_timeseries_from_bq(days, limit)
    except Exception as e:
        logger.error(f"Failed to fetch scoring timeseries: {e}")
        return {"ok": False, "reason": str(e)}


# ============================================================================
# CURATED FEEDS - Purpose-driven content collections
# ============================================================================

@app.get("/api/feeds")
async def list_feeds():
    """
    List all available curated feeds with metadata.

    Returns feed IDs, names, descriptions, and use cases.
    """
    return {
        "feeds": [
            {"id": k, **v} 
            for k, v in FeedDefinitions.FEEDS.items()
        ],
        "total": len(FeedDefinitions.FEEDS)
    }


@app.get("/api/feeds/{feed_id}")
async def get_feed(feed_id: str, limit: int = 20, min_score: int = 60):
    """
    Get articles from a specific curated feed.

    Args:
        feed_id: One of: daily_reads, trending, water_cooler, tutorials,
                 deep_dives, case_studies, this_week
        limit: Maximum articles to return (default 20)
        min_score: Minimum final score for feeds that use it (default 60)
    """
    feed = FeedDefinitions.FEEDS.get(feed_id)

    if not feed:
        raise HTTPException(
            status_code=404,
            detail=f"Feed '{feed_id}' not found. Available: {list(FeedDefinitions.FEEDS.keys())}"
        )

    try:
        # Get query with properly formatted table name
        query = FeedDefinitions.get_query(feed_id)

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("min_score", "INT64", min_score),
            ]
        )

        articles = []
        # Use simple query execution since prompts are trusted/static
        for row in bq_client.query(query, job_config=job_config).result():
            # Safe dict conversion for reliable access
            row_dict = dict(row)
            article = {
                "article_id": row_dict.get('article_id'),
                "title": row_dict.get('title'),
                "url": row_dict.get('url'),
                "summary": row_dict.get('summary'),
                "scores": {
                    "ai_relevance": row_dict.get('ai_relevance_score', 0),
                    "community": row_dict.get('community_score', 0),
                    "final": row_dict.get('final_score', 0),
                    "category": row_dict.get('score_category')
                },
                "actionability": row_dict.get('actionability'),
                "content_type": row_dict.get('content_type'),
                "estimated_read_time": row_dict.get('estimated_read_time'),
                "social_proof": {
                    "hn_points": row_dict.get('hn_points', 0),
                    "hn_comments": row_dict.get('hn_comments', 0),
                    "hn_url": row_dict.get('hn_url'),
                    "lobsters_points": row_dict.get('lobsters_points', 0),
                    "lobsters_comments": row_dict.get('lobsters_comments', 0),
                    "lobsters_url": row_dict.get('lobsters_url'),
                    "is_trending": row_dict.get('is_trending', False),
                    "is_viral": row_dict.get('is_viral', False),
                },
                "topics": row_dict.get('key_topics', []),
                "ai_reasoning": row_dict.get('ai_reasoning'),
                "category": row_dict.get('category'),
                "scored_at": row.scored_at.isoformat() if hasattr(row, 'scored_at') and row.scored_at else None
            }
            articles.append(article)

        return {
            "feed": {
                "id": feed_id,
                "name": feed['name'],
                "description": feed['description'],
                "purpose": feed['purpose'],
                "icon": feed['icon'],
                "use_case": feed['use_case'],
            },
            "count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Error fetching feed '{feed_id}': {e}")
        # In production, hide specific DB errors
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feeds/tutorials/topics")
async def get_tutorial_topics():
    """
    Get available tutorial topics with article counts.

    Helps users discover what tutorials are available.
    """
    try:
        query = f"""
        SELECT
            topic,
            COUNT(*) as article_count
        FROM `{PROJECT_ID}.{DATASET_ID}.articles_scored`,
        UNNEST(key_topics) as topic
        WHERE DATE(scored_at) >= CURRENT_DATE() - 14
          AND content_type = 'tutorial'
          AND actionability = 'high'
        GROUP BY topic
        ORDER BY article_count DESC
        LIMIT 20
        """

        topics = []
        for row in bq_client.query(query).result():
            topics.append({
                "topic": row.topic,
                "article_count": row.article_count
            })

        return {
            "feed": "tutorials",
            "topics": topics
        }

    except Exception as e:
        logger.error(f"Error fetching tutorial topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints with API key authentication (Secret Manager)
try:
    _admin_secret_name = os.getenv("ADMIN_API_KEY_SECRET_NAME", "ADMIN_API_KEY")
    ADMIN_API_KEY = get_secret_value(_admin_secret_name, PROJECT_ID)
except Exception as exc:
    ADMIN_API_KEY = None
    logger.warning("ADMIN_API_KEY not configured - admin endpoints disabled: %s", exc)

def verify_admin_key(request: Request) -> bool:
    """Verify admin API key from Authorization header."""
    if not ADMIN_API_KEY:
        logger.warning("ADMIN_API_KEY not configured - admin endpoints disabled")
        raise HTTPException(status_code=503, detail="Admin endpoints not configured")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_key = auth_header[7:]  # Strip "Bearer "
    if provided_key != ADMIN_API_KEY:
        logger.warning(f"Invalid admin API key attempt from {request.client.host}")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True

@app.post("/api/admin/reload-feeds")
async def reload_feeds_from_opml(request: Request):
    """
    Reload feeds from OPML file.

    Requires: Authorization: Bearer <ADMIN_API_KEY>
    """
    # Verify admin authentication
    verify_admin_key(request)

    try:
        # Import and run the loader
        from scripts.load_feeds_from_opml import load_feeds_to_bigquery

        opml_path = "data/feeds.opml"
        if not os.path.exists(opml_path):
            raise HTTPException(status_code=404, detail="OPML file not found")

        load_feeds_to_bigquery(opml_path, PROJECT_ID)

        return {
            "status": "success",
            "message": "Feeds reloaded from OPML",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"OPML file not found: {e}")
        raise HTTPException(status_code=404, detail="OPML file not found")
    except Exception as e:
        logger.error(f"Error reloading feeds: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload feeds")


# ============================================================================
# AI ASSISTANT - StruQ (Structured Query) Implementation
# ============================================================================

class AdminActionRequest(BaseModel):
    action: str

class VideoProcessRequest(BaseModel):
    url: str
    mock: bool = False
    mode: str = "comprehensive"

@app.post("/api/admin/process-video")
async def process_video_endpoint(
    request_body: VideoProcessRequest,
    authorization: str = Header(None)
):
    """Admin endpoint to trigger video processing."""
    # Re-using verify_admin_key for simplicity, assuming it handles the Bearer token.
    # If `verify_admin_key` expects a `Request` object, we might need a wrapper or
    # to pass the token directly. For now, let's adapt it slightly or assume it's fine.
    # A more robust solution would be a FastAPI dependency.
    if not ADMIN_API_KEY:
        logger.warning("ADMIN_API_KEY not configured - admin endpoints disabled")
        raise HTTPException(status_code=503, detail="Admin endpoints not configured")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_key = authorization[7:]  # Strip "Bearer "
    if provided_key != ADMIN_API_KEY:
        logger.warning(f"Invalid admin API key attempt for video processing")
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Run in background to avoid timeout? For now, run sync or return job ID.
    # Given Cloud Run timeouts (60m possible, but default 5m), let's try direct.
    # Ideally this should be a background task (Cloud Tasks).
    # For Hackathon: We'll run it and hope it finishes or user accepts timeout.
    
    try:
        article_id = lecture_service.process_video(request_body.url, mock=request_body.mock, mode=request_body.mode)
        if not article_id:
            raise HTTPException(status_code=500, detail="Video processing failed to produce an ID.")
        
        return {"message": "Video processed successfully", "article_id": article_id}
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lectures")
async def get_lectures():
    """Get list of processed video lectures."""
    try:
        lectures = lecture_service.get_all_lectures()
        return {"lectures": lectures}
    except Exception as e:
        logger.error(f"Failed to fetch lectures: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch lectures")

@app.get("/api/lectures/{article_id}")
async def get_lecture_detail(article_id: str):
    """Get single lecture details."""
    try:
        lecture = lecture_service.get_lecture_by_id(article_id)
        if not lecture:
            raise HTTPException(status_code=404, detail="Lecture not found")
        debug = lecture.pop("_debug_normalize", None)
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
        }
        if debug:
            headers["X-DevCaddie-Lecture-Normalize"] = f"{debug.get('path','?')}:{int(bool(debug.get('ok')))}"
        return JSONResponse(content=jsonable_encoder(lecture), headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch lecture {article_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch lecture")

@app.get("/lectures/{article_id}")
async def lecture_page(article_id: str):
    """Serve specific lecture page."""
    return FileResponse("static/lecture.html")

class UserQuery(BaseModel):
    query: str

@app.post("/api/assistant/query")
@limiter.limit("5/minute")
async def query_assistant(user_query: UserQuery, request: Request):
    """
    StruQ Assistant Endpoint.
    1. Validates input (InputValidator)
    2. Extracts structured intent (Vertex AI) - NO SQL GENERATION
    3. Maps to safe parameterized SQL
    4. Executes query
    """
    query_text = user_query.query
    ip_address = request.client.host

    # 0. Check Vacation Mode (Ultra-strict limits)
    # Note: check_vacation_limits raises HTTPException if limits exceeded
    if Config.is_vacation_mode():
        VacationLimiter.check_vacation_limits(ip_address)

    # 1. Validate Input
    try:
        # 0. Enforce Budget Guard (Strict Billing Limit)
        BudgetGuard.check_budget(ip_address)
        
        # 1. Validate Input
        InputValidator.validate_user_query(query_text)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid input")

    try:
        # 2. Extract Intent (StruQ)
        # Using native async call + Custom Tracing Span
        with tracer.start_as_current_span("ai_assistant_processing") as span:
            span.set_attribute("ai.operation", "extract_intent")
            intent = await content_assistant.analyze_intent(query_text)
        
        # RECORD USAGE (Update Budget)
        BudgetGuard.record_request(
            ip_address=ip_address,
            cost=BudgetGuard.COST_PER_REQUEST_USD
        )

        # Record vacation mode usage (if active)
        if Config.is_vacation_mode():
            VacationLimiter.record_vacation_request(ip_address)
        
        # 2b. Validate Output (Defense in Depth)
        # Ensure the LLM didn't slip through any PII or malicious patterns in the JSON
        if OutputValidator.contains_pii(str(intent.dict())):
             logger.warning(f"PII detected in intent: {intent}")
             # We might choose to block or redact. For now, we log.


        # 2c. Handle greeting intent (no SQL needed)
        if intent.intent_type == "greeting":
            return {
                "intent": intent.model_dump(),
                "count": 0,
                "articles": [],
                "message": "Hello! I'm your Dev Caddie assistant. Ask me about articles on topics like LLMs, Kubernetes, Airflow, Python, or request 'top 5 articles this week'."
            }

        # 2d. Handle Empty Search (Defense in Depth / UX)
        # If fallback occurs (e.g. injection blocked) or user input is vague, return empty.
        if intent.intent_type == "search" and not intent.topics:
             return {
                "intent": intent.model_dump(),
                "count": 0,
                "articles": [],
                "message": "I didn't detect any specific topics. Please ask about a technology like 'Spark', 'Python', or 'Iceberg'."
            }

        # 3. Build Safe Query
        query_components = content_assistant.build_query(
            intent, 
            project_id=PROJECT_ID, 
            dataset_id=DATASET_ID
        )
        
        # 4. Execute Query
        # Construct parameters list for BigQuery client
        bq_params = []
        for p in query_components["params"]:
            bq_params.append(
                bigquery.ScalarQueryParameter(p["name"], p["type"], p["value"])
            )
            
        job_config = bigquery.QueryJobConfig(query_parameters=bq_params)
        
        # Execute in thread pool (BigQuery is sync)
        with tracer.start_as_current_span("bigquery_execution") as span:
            span.set_attribute("db.system", "bigquery")
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(bq_client.query(query_components["query"], job_config=job_config).result())
            )
        
        # Format results
        articles = []
        for row in results:
            articles.append({
                "article_id": row.article_id,
                "title": row.title,
                "url": row.url,
                "summary": row.summary,
                "scores": {
                    "ai_relevance": row.ai_relevance_score,
                    "community": row.community_score,
                    "final": row.final_score
                },
                "content": {
                    "type": row.content_type,
                    "read_time": row.estimated_read_time
                },
                "topics": row.key_topics,
                "category": row.category,
                "scored_at": row.scored_at.isoformat() if row.scored_at else None
            })
            
        return {
            "intent": query_components["intent_debug"],
            "count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail="Assistant currently unavailable")


# ============================================================================
# MORNING BRIEFING - Gemini 2.0 Flash Script + Live Audio
# ============================================================================

@app.get("/api/briefing/today")
async def get_briefing_script():
    """
    Get today's AI-generated morning briefing script.
    Fallback: Use most recent available script if today's is missing.
    """
    local_dt = datetime.now(ZoneInfo("America/Denver"))
    utc_dt = datetime.now(timezone.utc)
    requested_date = local_dt.strftime("%Y-%m-%d")
    script_date = requested_date
    fallback_used = False

    script = await briefing_service.get_todays_script()
    if not script:
        latest = await briefing_service.get_latest_script()
        if latest and latest.get("script"):
            script = latest["script"]
            latest_date = latest.get("date")
            script_date = latest_date.isoformat() if hasattr(latest_date, "isoformat") else str(latest_date)
            fallback_used = True
        else:
            script = "No significant updates today."

    return {
        "date": script_date,
        "requested_date": requested_date,
        "datetime": local_dt.isoformat(),
        "datetime_utc": utc_dt.isoformat(),
        "timezone": "America/Denver",
        "script": script,
        "fallback": fallback_used,
        "vacation_mode": Config.is_vacation_mode(),
        "vacation_end_date": Config.VACATION_END_DATE
    }


@app.get("/api/briefing/top")
async def get_briefing_top_articles():
    """
    Top scored articles using the same query as the briefing service.
    """
    articles = await briefing_service.get_recent_top_articles(limit=5)
    return {"items": articles}


@app.post("/api/briefing/start")
async def start_briefing(request: Request, force: bool = False):
    """
    Create a Daily room + token and start the Pipecat briefing bot.
    Returns room URL + token for the frontend to join.
    """
    if Config.is_vacation_mode() and not force:
        raise HTTPException(status_code=409, detail="Vacation mode enabled. Briefing is paused.")
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")[:160]
    cache_key = hashkey(client_ip, user_agent)
    secret_name = os.getenv("DAILY_API_KEY_SECRET_NAME", "DAILY_API_KEY")
    daily_api_key = get_secret_value(secret_name, PROJECT_ID)
    bot_start_url = os.getenv("BOT_START_URL")

    # Fetch briefing script to pass to the bot
    script = await briefing_service.get_todays_script()
    if not script:
        latest = await briefing_service.get_latest_script()
        if latest and latest.get("script"):
            script = latest["script"]
        else:
            script = "No significant updates today."

    def _start_bot_sidecar(room_url: str, bot_token: str, script_text: str | None) -> bool:
        """Best-effort start of VM sidecar from Cloud Run."""
        try:
            payload = {"room_url": room_url, "token": bot_token, "script": script_text}
            resp = requests.post(f"{bot_start_url.rstrip('/')}/start", json=payload, timeout=10)
            if resp.status_code >= 400:
                logger.warning("Sidecar start failed: %s %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:
            logger.warning("Sidecar start exception: %s", exc)
            return False

    # Refresh safety: if a cached room exists, tear it down before creating a new one
    cached = briefing_start_cache.pop(cache_key, None)
    if cached:
        try:
            room_name = cached.get("room_name")
            if room_name:
                requests.delete(
                    f"https://api.daily.co/v1/rooms/{room_name}",
                    headers={"Authorization": f"Bearer {daily_api_key}"},
                    timeout=10,
                )
        except Exception as exc:
            logger.warning("Briefing start: failed to delete cached room: %s", exc)
        try:
            if bot_start_url and cached.get("room_url"):
                requests.post(
                    f"{bot_start_url.rstrip('/')}/stop",
                    json={"room_url": cached["room_url"]},
                    timeout=10,
                )
        except Exception as exc:
            logger.warning("Briefing start: failed to stop sidecar for cached room: %s", exc)

    # Create a short-lived room (30 minutes)
    exp = int(time.time()) + 1800
    room_resp = requests.post(
        "https://api.daily.co/v1/rooms",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "privacy": "private",
            "properties": {
                "exp": exp,
                "enable_chat": False,
                "enable_screenshare": False,
                "start_video_off": True,
                "start_audio_off": False
            }
        },
        timeout=20,
    )
    room_resp.raise_for_status()
    room = room_resp.json()

    room_name = room.get("name")
    room_url = room.get("url")
    if not room_name or not room_url:
        raise HTTPException(status_code=500, detail="Daily room creation failed")

    # Create a token for the user
    token_resp = requests.post(
        "https://api.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={"properties": {"room_name": room_name}},
        timeout=20,
    )
    token_resp.raise_for_status()
    user_token = token_resp.json().get("token")
    if not user_token:
        raise HTTPException(status_code=500, detail="Daily token creation failed")

    # Create a token for the bot (owner)
    bot_token_resp = requests.post(
        "https://api.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "room_name": room_name,
                "is_owner": True,
                "start_audio_off": False,
                "start_video_off": True,
            }
        },
        timeout=20,
    )
    bot_token_resp.raise_for_status()
    bot_token = bot_token_resp.json().get("token")
    if not bot_token:
        raise HTTPException(status_code=500, detail="Daily bot token creation failed")

    if not bot_start_url:
        raise HTTPException(status_code=500, detail="BOT_START_URL not configured")

    briefing_start_cache[cache_key] = {
        "room_name": room_name,
        "room_url": room_url,
    }
    logger.info(
        "Briefing start: new room=%s for client=%s ua=%s",
        room_name,
        client_ip,
        user_agent
    )

    bot_started = _start_bot_sidecar(room_url, bot_token, script)
    return {
        "room_url": room_url,
        "token": user_token,
        "bot_token": bot_token,
        "bot_start_url": bot_start_url,
        "bot_started": bot_started,
        "script": script
    }


@app.post("/api/briefing/stop")
async def stop_briefing(request: Request):
    """
    Clear cached briefing room for this client so next start creates a new room.
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")[:160]
    cache_key = hashkey(client_ip, user_agent)
    removed = briefing_start_cache.pop(cache_key, None)
    if removed:
        try:
            secret_name = os.getenv("DAILY_API_KEY_SECRET_NAME", "DAILY_API_KEY")
            daily_api_key = get_secret_value(secret_name, PROJECT_ID)
            room_name = removed.get("room_name")
            if room_name:
                requests.delete(
                    f"https://api.daily.co/v1/rooms/{room_name}",
                    headers={"Authorization": f"Bearer {daily_api_key}"},
                    timeout=10,
                )
        except Exception as exc:
            logger.warning("Briefing stop: failed to delete Daily room: %s", exc)

        try:
            bot_start_url = os.getenv("BOT_START_URL")
            if bot_start_url and removed.get("room_url"):
                requests.post(
                    f"{bot_start_url.rstrip('/')}/stop",
                    json={"room_url": removed["room_url"]},
                    timeout=10,
                )
        except Exception as exc:
            logger.warning("Briefing stop: failed to stop sidecar: %s", exc)
    logger.info(
        "Briefing stop: cache_clear=%s for client=%s ua=%s",
        bool(removed),
        client_ip,
        user_agent,
    )
    return {"cleared": bool(removed)}


@app.get("/api/briefing/observability/kpis")
async def briefing_observability_kpis(days: int = 7):
    try:
        return await fetch_briefing_kpis_from_bq(days)
    except Exception as exc:
        logger.warning("Briefing observability kpis failed: %s", exc)
        return {"ok": False, "reason": "bq-error"}


@app.get("/api/briefing/observability/timeseries")
async def briefing_observability_timeseries(limit: int = 200, days: int = 7):
    try:
        return await fetch_briefing_timeseries_from_bq(days, limit)
    except Exception as exc:
        logger.warning("Briefing observability timeseries failed: %s", exc)
        return {"ok": False, "reason": "bq-error"}

@app.websocket("/ws/briefing/live")
async def websocket_briefing(websocket: WebSocket):
    """
    Real-time Audio Briefing via Gemini Live 2.5 Flash.
    """
    await websocket.accept()
    await briefing_service.handle_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
