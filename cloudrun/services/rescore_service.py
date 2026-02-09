"""
Re-scoring Service for Rolling Community Scores

Articles can go viral days after initial scoring.
This service re-checks community scores on a decay schedule:
- First 48 hours: Check every 16 hours (3x)
- Days 2-7: Check daily
- Days 7-30: Check weekly
- After 30 days: Stop re-checking
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from services.community_scorer import CommunityScorer, calculate_final_score

logger = logging.getLogger(__name__)


@dataclass
class ArticleToRescore:
    """Article that needs community score update."""
    article_id: str
    url: str
    ai_relevance_score: int
    current_community_score: int
    current_final_score: int
    scored_at: datetime
    last_community_check: datetime
    community_check_count: int


@dataclass
class RescoreResult:
    """Result of re-scoring an article."""
    article_id: str
    old_community_score: int
    new_community_score: int
    old_final_score: int
    new_final_score: int
    is_trending: bool
    is_viral: bool
    confidence: str
    score_changed: bool
    newly_viral: bool  # Just crossed viral threshold


class RescoreService:
    """
    Service to re-check and update community scores.

    Implements decay schedule:
    - Aggressive checking for new articles (48h)
    - Daily for first week
    - Weekly for first month
    - Stop after 30 days
    """

    def __init__(self, bigquery_client=None):
        """Initialize re-score service."""
        self.community_scorer = CommunityScorer()
        self.bq_client = bigquery_client

    def get_articles_for_rescore(self) -> List[ArticleToRescore]:
        """
        Get articles that need community score re-checking.

        Returns articles based on decay schedule.
        """
        if not self.bq_client:
            logger.warning("No BigQuery client - returning empty list")
            return []

        query = """
        SELECT
            article_id,
            url,
            ai_relevance_score,
            community_score as current_community_score,
            final_score as current_final_score,
            scored_at,
            last_community_check,
            community_check_count
        FROM `content_intelligence.articles_scored`
        WHERE
            -- First 48 hours: check every 16 hours
            (scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 48 HOUR)
             AND last_community_check < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 16 HOUR))
            OR
            -- Days 2-7: check daily
            (scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
             AND scored_at < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 48 HOUR)
             AND last_community_check < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR))
            OR
            -- Days 7-30: check weekly
            (scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
             AND scored_at < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
             AND last_community_check < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY))
        ORDER BY scored_at DESC
        LIMIT 500
        """

        try:
            results = self.bq_client.query(query).result()

            articles = []
            for row in results:
                articles.append(ArticleToRescore(
                    article_id=row.article_id,
                    url=row.url,
                    ai_relevance_score=row.ai_relevance_score,
                    current_community_score=row.current_community_score,
                    current_final_score=row.current_final_score,
                    scored_at=row.scored_at,
                    last_community_check=row.last_community_check,
                    community_check_count=row.community_check_count
                ))

            logger.info(f"Found {len(articles)} articles for re-scoring")
            return articles

        except Exception as e:
            logger.error(f"Error fetching articles for rescore: {e}")
            return []

    def rescore_article(self, article: ArticleToRescore) -> RescoreResult:
        """
        Re-check community score for a single article.

        Args:
            article: Article to re-score

        Returns:
            RescoreResult with updated scores
        """
        # Get fresh community score
        community_data = self.community_scorer.score_article(article.url)

        # Calculate new final score
        final_data = calculate_final_score(
            ai_relevance=article.ai_relevance_score,
            community_score=community_data.total_score,
            confidence=community_data.confidence
        )

        # Check if score changed significantly (>5 points)
        score_changed = abs(community_data.total_score - article.current_community_score) > 5

        # Check if newly viral (wasn't viral before, is now)
        was_viral = article.current_community_score >= 70  # Approximate
        newly_viral = community_data.is_viral and not was_viral

        return RescoreResult(
            article_id=article.article_id,
            old_community_score=article.current_community_score,
            new_community_score=community_data.total_score,
            old_final_score=article.current_final_score,
            new_final_score=final_data['final_score'],
            is_trending=community_data.is_trending,
            is_viral=community_data.is_viral,
            confidence=community_data.confidence,
            score_changed=score_changed,
            newly_viral=newly_viral
        )

    def update_article_score(self, result: RescoreResult) -> bool:
        """
        Update article score in BigQuery.

        Args:
            result: RescoreResult with new scores

        Returns:
            True if update successful
        """
        if not self.bq_client:
            logger.warning("No BigQuery client - skipping update")
            return False

        query = """
        UPDATE `content_intelligence.articles_scored`
        SET
            community_score = @new_community_score,
            final_score = @new_final_score,
            is_trending = @is_trending,
            is_viral = @is_viral,
            community_confidence = @confidence,
            last_community_check = CURRENT_TIMESTAMP(),
            community_check_count = community_check_count + 1,
            score_history = ARRAY_CONCAT(
                IFNULL(score_history, []),
                [STRUCT(
                    CURRENT_TIMESTAMP() as checked_at,
                    @new_community_score as community_score,
                    @new_final_score as final_score
                )]
            )
        WHERE article_id = @article_id
        """

        try:
            from google.cloud import bigquery

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("article_id", "STRING", result.article_id),
                    bigquery.ScalarQueryParameter("new_community_score", "INT64", result.new_community_score),
                    bigquery.ScalarQueryParameter("new_final_score", "INT64", result.new_final_score),
                    bigquery.ScalarQueryParameter("is_trending", "BOOL", result.is_trending),
                    bigquery.ScalarQueryParameter("is_viral", "BOOL", result.is_viral),
                    bigquery.ScalarQueryParameter("confidence", "STRING", result.confidence),
                ]
            )

            self.bq_client.query(query, job_config=job_config).result()

            logger.info(
                f"Updated article {result.article_id}: "
                f"community {result.old_community_score} -> {result.new_community_score}, "
                f"final {result.old_final_score} -> {result.new_final_score}"
            )

            return True

        except Exception as e:
            logger.error(f"Error updating article score: {e}")
            return False

    def run_rescore_batch(self, limit: int = 100) -> Dict:
        """
        Run a batch of re-scoring operations.

        Args:
            limit: Maximum articles to process

        Returns:
            Summary statistics
        """
        articles = self.get_articles_for_rescore()[:limit]

        stats = {
            'total_checked': 0,
            'scores_changed': 0,
            'newly_viral': 0,
            'now_trending': 0,
            'errors': 0
        }

        for article in articles:
            try:
                result = self.rescore_article(article)
                stats['total_checked'] += 1

                if result.score_changed:
                    stats['scores_changed'] += 1
                    self.update_article_score(result)

                if result.newly_viral:
                    stats['newly_viral'] += 1
                    logger.info(f"Article went viral: {article.url}")

                if result.is_trending:
                    stats['now_trending'] += 1

            except Exception as e:
                logger.error(f"Error re-scoring {article.url}: {e}")
                stats['errors'] += 1

        logger.info(f"Rescore batch complete: {stats}")
        return stats


def should_rescore(scored_at: datetime, last_check: datetime) -> bool:
    """
    Determine if an article should be re-scored based on decay schedule.

    Args:
        scored_at: When article was first scored
        last_check: When community score was last checked

    Returns:
        True if article should be re-scored
    """
    now = datetime.utcnow()
    age = now - scored_at
    since_last_check = now - last_check

    # First 48 hours: check every 16 hours
    if age <= timedelta(hours=48):
        return since_last_check >= timedelta(hours=16)

    # Days 2-7: check daily
    if age <= timedelta(days=7):
        return since_last_check >= timedelta(days=1)

    # Days 7-30: check weekly
    if age <= timedelta(days=30):
        return since_last_check >= timedelta(days=7)

    # After 30 days: no more re-scoring
    return False


if __name__ == "__main__":
    # Test decay schedule
    from datetime import datetime, timedelta

    now = datetime.utcnow()

    test_cases = [
        ("1 hour old, never checked", now - timedelta(hours=1), now - timedelta(hours=1)),
        ("1 hour old, checked 20h ago", now - timedelta(hours=1), now - timedelta(hours=20)),
        ("3 days old, checked 2h ago", now - timedelta(days=3), now - timedelta(hours=2)),
        ("3 days old, checked 25h ago", now - timedelta(days=3), now - timedelta(hours=25)),
        ("10 days old, checked 2 days ago", now - timedelta(days=10), now - timedelta(days=2)),
        ("10 days old, checked 8 days ago", now - timedelta(days=10), now - timedelta(days=8)),
        ("60 days old", now - timedelta(days=60), now - timedelta(days=30)),
    ]

    print("Re-scoring decay schedule tests:")
    print("-" * 60)
    for name, scored_at, last_check in test_cases:
        result = should_rescore(scored_at, last_check)
        print(f"{name}: {'YES' if result else 'NO'}")
