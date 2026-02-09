import os

# Get project ID from environment
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = "content_intelligence"
TABLE_ID = "articles_scored"

# Fully qualified table name
ARTICLES_TABLE = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"


class FeedDefinitions:
    """Complete feed definitions with clear purposes."""

    # Map friendly IDs to the definitions
    # Note: Queries use {table} placeholder, replaced at runtime
    FEEDS = {
        'daily_reads': {  # 'For you' mapped to default daily
            'name': 'For You',
            'description': 'Personally relevant articles validated by the community',
            'purpose': 'Personally relevant articles validated by the community',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)
                    AND ai_relevance_score >= 70  -- High relevance
                    AND final_score >= 75  -- High quality
                ORDER BY final_score DESC
                LIMIT 20
            ''',
            'icon': '🎯',
            'use_case': 'Daily morning reading (10-15 min)'
        },

        'trending': {
            'name': 'Trending',
            'description': 'Currently popular technical content',
            'purpose': 'Currently popular technical content',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)
                    AND community_score >= 70
                    AND ai_relevance_score >= 25  -- Junk floor
                    AND (hn_points >= 300 OR is_trending = TRUE)
                ORDER BY community_score DESC, hn_points DESC
                LIMIT 20
            ''',
            'icon': '🔥',
            'use_case': 'Check 1-2x per day for hot topics'
        },

        'water_cooler': {
            'name': 'Water Cooler',
            'description': 'Thought-provoking essays and culture pieces',
            'purpose': 'Thought-provoking essays and culture pieces',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
                    AND (
                        lobsters_points >= 20  -- Lobsters-curated content
                        OR (hn_comments >= 100 AND content_type = 'opinion')
                    )
                    AND ai_relevance_score >= 10  -- Relaxed floor (allow culture pieces)
                ORDER BY lobsters_points DESC, hn_comments DESC
                LIMIT 15
            ''',
            'icon': '💬',
            'use_case': 'Weekend reading - culture and opinion'
        },

        'tutorials': {
            'name': 'Tutorials',
            'description': 'Hands-on, actionable how-to guides',
            'purpose': 'Hands-on, actionable how-to guides',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
                    AND ai_relevance_score >= 60
                    AND actionability = 'high'  -- Step-by-step guides
                    AND content_type = 'tutorial'  -- Explicitly tutorials
                    AND estimated_read_time IN ('quick', 'medium')  -- 5-15 min
                ORDER BY ai_relevance_score DESC, community_score DESC
                LIMIT 20
            ''',
            'icon': '🛠️',
            'use_case': 'When you need to build/implement something'
        },

        'deep_dives': {
            'name': 'Deep Dives',
            'description': 'Long-form technical content for focused learning',
            'purpose': 'Long-form technical content for focused learning',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
                    AND ai_relevance_score >= 70
                    AND (
                        content_type = 'deep_dive'  -- Trust AI classification (even if fresh/low score)
                        OR (
                            estimated_read_time IN ('long', 'deep')
                            AND community_score >= 50
                            AND content_type NOT IN ('news', 'opinion')
                        )
                    )
                ORDER BY
                    CASE estimated_read_time
                        WHEN 'deep' THEN 1
                        WHEN 'long' THEN 2
                        ELSE 3
                    END,
                    ai_relevance_score DESC
                LIMIT 20
            ''',
            'icon': '📚',
            'use_case': 'Weekend deep learning sessions (30+ min)'
        },

        'case_studies': {
            'name': 'Case Studies',
            'description': 'Real-world implementations and lessons learned',
            'purpose': 'Real-world implementations and lessons learned',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 21 DAY)
                    AND ai_relevance_score >= 60
                    AND content_type = 'case_study'
                    AND community_score >= 40  -- Some validation
                ORDER BY community_score DESC, ai_relevance_score DESC
                LIMIT 20
            ''',
            'icon': '🏢',
            'use_case': 'Learn patterns from production systems'
        },

        'this_week': {
            'name': 'This Week',
            'description': 'All scored articles from the past week',
            'purpose': 'Weekly digest - catch anything you missed',
            'query': '''
                SELECT *
                FROM {table}
                WHERE
                    scored_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                    AND final_score >= 60
                ORDER BY final_score DESC
                LIMIT 50
            ''',
            'icon': '📊',
            'use_case': 'Weekend review of the week'
        }
    }

    @staticmethod
    def get_query(feed_id: str, project_id: str = None) -> str:
        """Get the SQL query for a specific feed ID.

        Args:
            feed_id: The feed identifier
            project_id: Optional project ID override. If not provided, uses PROJECT_ID env var.

        Returns:
            Formatted SQL query string, or None if feed not found.
        """
        feed = FeedDefinitions.FEEDS.get(feed_id)
        if not feed:
            return None

        # Use provided project_id or fall back to module-level constant
        pid = project_id or PROJECT_ID
        table = f"`{pid}.{DATASET_ID}.{TABLE_ID}`"

        return feed['query'].format(table=table)

    @staticmethod
    def get_metadata(feed_id: str) -> dict:
        """Get metadata for a specific feed ID."""
        feed = FeedDefinitions.FEEDS.get(feed_id)
        if not feed:
            return None
        return {
            'name': feed['name'],
            'description': feed['description'],
            'icon': feed['icon'],
            'purpose': feed['purpose'],
            'use_case': feed['use_case']
        }

    @staticmethod
    def list_feeds() -> list:
        """List all available feed IDs with their metadata."""
        return [
            {
                'id': feed_id,
                **FeedDefinitions.get_metadata(feed_id)
            }
            for feed_id in FeedDefinitions.FEEDS.keys()
        ]
