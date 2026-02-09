"""
Community Score Calculator

Aggregates social signals from HackerNews and Lobste.rs to determine
if an article is popular/valuable in the tech community.

Two-dimensional scoring:
- Dimension 1: AI Relevance (personal match)
- Dimension 2: Community Score (social proof)
"""

import requests
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CommunityScore:
    """Community engagement metrics for an article."""
    total_score: int = 0  # Combined score (0-100)

    # HackerNews metrics
    hn_points: int = 0
    hn_comments: int = 0
    hn_url: Optional[str] = None
    hn_posted_at: Optional[str] = None

    # Lobste.rs metrics
    lobsters_points: int = 0
    lobsters_comments: int = 0
    lobsters_url: Optional[str] = None

    # Metadata
    is_trending: bool = False
    is_viral: bool = False
    confidence: str = "low"  # low, medium, high
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CommunityScorer:
    """
    Calculate community engagement scores for articles.

    Uses free APIs:
    - HackerNews Algolia API (primary)
    - Lobste.rs API (secondary)
    """

    def __init__(self, timeout: int = 5):
        """Initialize community scorer."""
        self.hn_api = "https://hn.algolia.com/api/v1/search"
        self.lobsters_api = "https://lobste.rs/search.json"
        self.timeout = timeout
        self.lobsters_cache = None # Cache for feed-based lookup

    def score_article(self, article_url: str) -> CommunityScore:
        """
        Calculate community score for an article.

        Args:
            article_url: URL of the article

        Returns:
            CommunityScore with aggregated metrics
        """
        # Get scores from each platform
        hn_data = self._get_hn_score(article_url)
        lobsters_data = self._get_lobsters_score(article_url)

        # Calculate weighted total score (0-100)
        total_score = self._calculate_total_score(hn_data, lobsters_data)

        # Determine trending/viral status
        is_trending = self._is_trending(hn_data, lobsters_data)
        is_viral = self._is_viral(hn_data, lobsters_data)

        # Confidence level
        confidence = self._calculate_confidence(hn_data, lobsters_data)

        return CommunityScore(
            total_score=total_score,
            hn_points=hn_data.get('points', 0),
            hn_comments=hn_data.get('comments', 0),
            hn_url=hn_data.get('hn_url'),
            hn_posted_at=hn_data.get('created_at'),
            lobsters_points=lobsters_data.get('points', 0),
            lobsters_comments=lobsters_data.get('comments', 0),
            lobsters_url=lobsters_data.get('url'),
            is_trending=is_trending,
            is_viral=is_viral,
            confidence=confidence
        )

    def _get_hn_score(self, url: str) -> Dict:
        """Get HackerNews score for article."""
        try:
            # 1. Strip query params for better matching (Aggressive Normalization)
            # Simple strip of '?' and everything after, as HN usually links canonical URLs
            search_url = url.split('?')[0]
            
            response = requests.get(
                self.hn_api,
                params={'query': search_url, 'tags': 'story', 'hitsPerPage': 1},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if data.get('hits'):
                hit = data['hits'][0]
                return {
                    'points': hit.get('points', 0),
                    'comments': hit.get('num_comments', 0),
                    'hn_url': f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    'created_at': hit.get('created_at')
                }

        except requests.exceptions.Timeout:
            logger.warning(f"HN API timeout for {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching HN score for {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching HN score: {e}")

        return {'points': 0, 'comments': 0, 'hn_url': None, 'created_at': None}

    def _get_lobsters_score(self, url: str) -> Dict:
        """
        Get Lobste.rs score for article using Feed-Based Lookup.
        
        To avoid N+1 queries, we fetch the 'hottest' and 'newest' feeds once
        and lookup the URL in memory.
        """
        if self.lobsters_cache is None:
            self._refresh_lobsters_cache()

        # Normalize URL for matching (strip trailing slash)
        normalized_url = url.rstrip('/')
        
        if normalized_url in self.lobsters_cache:
            data = self.lobsters_cache[normalized_url]
            return {
                'points': data.get('score', 0),
                'comments': data.get('comment_count', 0),
                'url': data.get('short_id_url') or data.get('comments_url')
            }
            
        return {'points': 0, 'comments': 0, 'url': None}

    def _refresh_lobsters_cache(self):
        """Fetch hottest and newest feeds from Lobsters to build a lookup cache."""
        self.lobsters_cache = {}
        feeds = [
            "https://lobste.rs/hottest.json",
            "https://lobste.rs/newest.json"
        ]
        
        import time
        
        # User Agent is critical for Lobsters
        headers = {
            'User-Agent': "ContentIntelligenceHub/1.0 (+mailto:your-email@example.com)"
        }

        for feed_url in feeds:
            try:
                response = requests.get(feed_url, headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        # Store by URL
                        url = item.get('url', '').rstrip('/')
                        if url:
                            self.lobsters_cache[url] = item
                
                # Respect rate limits
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error fetching Lobsters feed {feed_url}: {e}")
                # Continue to next feed even if one fails


    def _calculate_total_score(self, hn_data: Dict, lobsters_data: Dict) -> int:
        """
        Calculate total community score (0-100).

        Scoring formula:
        - HN points: weighted 70% (larger platform)
        - Lobsters points: weighted 20%
        - Comments: weighted 10% (engagement indicator)

        Thresholds:
        - 500+ HN points = max HN score
        - 100+ Lobsters points = max Lobsters score
        - 100+ total comments = max comment score
        """
        hn_points = hn_data.get('points', 0)
        hn_comments = hn_data.get('comments', 0)
        lobsters_points = lobsters_data.get('points', 0)
        lobsters_comments = lobsters_data.get('comments', 0)

        # Normalize scores (cap at reasonable maximums)
        hn_score = min(hn_points / 500.0 * 70, 70)
        lobsters_score = min(lobsters_points / 100.0 * 20, 20)
        comment_score = min((hn_comments + lobsters_comments) / 100.0 * 10, 10)

        total = int(hn_score + lobsters_score + comment_score)

        return min(total, 100)

    def _is_trending(self, hn_data: Dict, lobsters_data: Dict) -> bool:
        """
        Determine if article is currently trending.

        Trending = 100+ HN points OR 50+ total comments
        """
        hn_points = hn_data.get('points', 0)
        total_comments = hn_data.get('comments', 0) + lobsters_data.get('comments', 0)

        return hn_points >= 100 or total_comments >= 50

    def _is_viral(self, hn_data: Dict, lobsters_data: Dict) -> bool:
        """
        Determine if article has gone viral.

        Viral = 500+ HN points OR 200+ comments
        """
        hn_points = hn_data.get('points', 0)
        total_comments = hn_data.get('comments', 0) + lobsters_data.get('comments', 0)

        return hn_points >= 500 or total_comments >= 200

    def _calculate_confidence(self, hn_data: Dict, lobsters_data: Dict) -> str:
        """
        Calculate confidence level in community score.

        - high: On multiple platforms with engagement
        - medium: On one platform with engagement
        - low: No or minimal social signals
        """
        signals = 0

        if hn_data.get('points', 0) > 10:
            signals += 1
        if lobsters_data.get('points', 0) > 5:
            signals += 1

        if signals >= 2:
            return "high"
        elif signals == 1:
            return "medium"
        else:
            return "low"


def calculate_final_score(
    ai_relevance: int,
    community_score: int,
    confidence: str
) -> Dict:
    """
    Calculate final composite score combining AI and community signals.

    Weighting strategy:
    - High confidence: 50% AI, 50% community
    - Medium confidence: 70% AI, 30% community
    - Low confidence: 90% AI, 10% community (trust AI more)

    Viral Override:
    - If community_score >= 70: flip to 30% AI, 70% community
    - Rationale: Viral content is worth seeing even if AI thinks it's off-topic

    Args:
        ai_relevance: AI relevance score (0-100)
        community_score: Community score (0-100)
        confidence: Confidence level (low, medium, high)

    Returns:
        Dict with final_score and category
    """
    weights = {
        'high': (0.5, 0.5),
        'medium': (0.7, 0.3),
        'low': (0.9, 0.1)
    }

    ai_weight, community_weight = weights.get(confidence, (0.9, 0.1))

    # Junk Floor: If AI scores it very low (<25), it's likely low-quality content
    # (link-only, rant, drama). Community hype shouldn't save junk.
    if ai_relevance < 25:
        final_score = ai_relevance  # No community boost for junk
    else:
        # Viral Override: If community score is high, favor community signal
        # BUT: Only if AI relevance is at least 25 (Veto Floor).
        # This prevents purely off-topic viral news (e.g. politics) from taking over.
        if community_score >= 70 and ai_relevance >= 25:
            ai_weight, community_weight = 0.3, 0.7

        final_score = int(ai_relevance * ai_weight + community_score * community_weight)

    # Determine category
    if final_score >= 80:
        category = "must-read"
    elif final_score >= 60:
        category = "recommended"
    else:
        category = "optional"

    return {
        'final_score': final_score,
        'ai_relevance': ai_relevance,
        'community_score': community_score,
        'confidence': confidence,
        'category': category
    }


if __name__ == "__main__":
    # Test community scorer
    scorer = CommunityScorer()

    # Test with a popular article
    test_url = "https://martinfowler.com/articles/data-monolith-to-mesh.html"
    score = scorer.score_article(test_url)

    print(f"URL: {test_url}")
    print(f"Community Score: {score.total_score}/100")
    print(f"HN: {score.hn_points} points, {score.hn_comments} comments")
    print(f"Lobsters: {score.lobsters_points} points, {score.lobsters_comments} comments")
    print(f"Trending: {score.is_trending}")
    print(f"Viral: {score.is_viral}")
    print(f"Confidence: {score.confidence}")

    # Test final score calculation
    final = calculate_final_score(
        ai_relevance=85,
        community_score=score.total_score,
        confidence=score.confidence
    )
    print(f"\nFinal Score: {final['final_score']}/100 ({final['category']})")
