# cloudrun/services/url_normalizer.py

"""
URL Normalization for Article ID Generation

Normalizes URLs to create consistent article_id hashes, preventing
duplicates when the same article appears with different URL variations.

Usage:
    from services.url_normalizer import normalize_url, generate_article_id

    article_id = generate_article_id(url)  # For dedup
    # Store original url in BigQuery for display/HN lookup
"""

import hashlib
import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


# Tracking parameters to strip (don't affect content)
TRACKING_PARAMS = {
    # Google Analytics
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    # Facebook
    'fbclid', 'fb_action_ids', 'fb_action_types', 'fb_source', 'fb_ref',
    # Google Ads
    'gclid', 'gclsrc', 'dclid',
    # Microsoft/Bing
    'msclkid',
    # Mailchimp
    'mc_cid', 'mc_eid',
    # Generic tracking
    'ref', 'source', 'via', 'campaign', 'medium',
    # Social
    'share', 'shared', 'si',
    # Other common
    '_ga', '_gl', 'trk', 'trkInfo', 'referer', 'referrer',
}

# Parameters that affect content (keep these)
CONTENT_PARAMS = {
    'id', 'p', 'page', 'article', 'post', 'story', 'v', 'video',
    'q', 'query', 'search', 's', 'tab', 'section', 'category',
}


def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistent article_id generation.

    Normalizations applied:
    - Lowercase scheme and domain
    - Remove www. prefix
    - Standardize to https://
    - Remove trailing slash (except for root)
    - Remove fragment (#...)
    - Remove tracking query parameters
    - Sort remaining query parameters

    Args:
        url: Original URL from RSS feed

    Returns:
        Normalized URL string

    Examples:
        >>> normalize_url("HTTP://WWW.Example.COM/Post/")
        'https://example.com/post'

        >>> normalize_url("https://blog.com/article?utm_source=twitter&id=123")
        'https://blog.com/article?id=123'
    """
    if not url:
        return ""

    # Parse the URL
    parsed = urlparse(url.strip())

    # Normalize scheme to https
    scheme = 'https'

    # Normalize domain: lowercase, remove www.
    netloc = parsed.netloc.lower()
    if netloc.startswith('www.'):
        netloc = netloc[4:]

    # Normalize path: lowercase, remove trailing slash (but keep root /)
    path = parsed.path.lower()
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')

    # Handle empty path
    if not path:
        path = '/'

    # Filter query parameters: remove tracking, keep content-affecting
    query_params = parse_qs(parsed.query, keep_blank_values=False)
    filtered_params = {}

    for key, values in query_params.items():
        key_lower = key.lower()
        # Keep parameter if it's a content param OR not a tracking param
        if key_lower in CONTENT_PARAMS or key_lower not in TRACKING_PARAMS:
            filtered_params[key] = values[0] if len(values) == 1 else values

    # Sort and encode query string
    if filtered_params:
        # Sort by key for consistency
        sorted_params = sorted(filtered_params.items())
        query = urlencode(sorted_params, doseq=True)
    else:
        query = ''

    # Reconstruct URL (no fragment)
    normalized = urlunparse((
        scheme,
        netloc,
        path,
        '',      # params (rarely used)
        query,
        ''       # fragment (removed)
    ))

    return normalized


def generate_article_id(url: str) -> str:
    """
    Generate a consistent article ID from a URL.

    Uses normalized URL to ensure same article with different
    URL variations gets the same ID.

    Args:
        url: Original URL from RSS feed

    Returns:
        MD5 hash of normalized URL (32 char hex string)

    Examples:
        >>> generate_article_id("https://example.com/post")
        'a1b2c3d4e5f6...'

        >>> # Same article, different URL format -> same ID
        >>> generate_article_id("http://www.example.com/post/")
        'a1b2c3d4e5f6...'  # Same as above
    """
    normalized = normalize_url(url)
    return hashlib.md5(normalized.encode()).hexdigest()


def urls_match(url1: str, url2: str) -> bool:
    """
    Check if two URLs refer to the same article.

    Args:
        url1: First URL
        url2: Second URL

    Returns:
        True if URLs normalize to the same value
    """
    return normalize_url(url1) == normalize_url(url2)
