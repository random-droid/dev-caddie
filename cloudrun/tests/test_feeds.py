import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime
import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from services.feed_definitions import FeedDefinitions

client = TestClient(app)

@pytest.fixture
def mock_bq_client():
    with patch('main.bq_client') as mock:
        yield mock

def test_list_feeds():
    """Verify we can list all available feeds."""
    response = client.get("/api/feeds")
    assert response.status_code == 200
    data = response.json()
    assert "feeds" in data
    assert data["total"] == 7
    
    # Check for specific feeds
    feed_ids = [f['id'] for f in data['feeds']]
    assert 'trending' in feed_ids
    assert 'water_cooler' in feed_ids
    assert 'daily_reads' in feed_ids

def test_get_trending_feed_query_logic(mock_bq_client):
    """
    Verify that the 'trending' feed uses the correct SQL logic.
    This specifically checks the USER's concern about 'rolling score' logic.
    """
    # Mock empty result for simplicity, we just want to check the query passed
    mock_job = MagicMock()
    mock_job.result.return_value = []
    mock_bq_client.query.return_value = mock_job

    response = client.get("/api/feeds/trending")
    assert response.status_code == 200

    # Get the actual call arguments
    args, _ = mock_bq_client.query.call_args
    query_executed = args[0]

    # VERIFY ROLLING WINDOW LOGIC
    # Expecting: DATE(scored_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)
    assert "DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)" in query_executed
    
    # VERIFY SCORE LOGIC
    # Expecting: community_score >= 70
    assert "community_score >= 70" in query_executed

def test_get_feed_response_structure(mock_bq_client):
    """Verify that BigQuery rows are correctly converted to API response."""
    
    # Create a mock row object
    class MockRow:
        def __init__(self):
            self.article_id = "test-123"
            self.title = "Test Article"
            self.url = "http://test.com"
            self.summary = "A summary"
            self.ai_relevance_score = 85
            self.community_score = 90
            self.final_score = 88
            self.score_category = "Must Read"
            self.actionability = "High"
            self.content_type = "tutorial"
            self.estimated_read_time = "medium"
            self.key_topics = ["Python", "Testing"]
            self.ai_reasoning = "Good content"
            self.category = "Data Engineering"
            self.scored_at = datetime.now()
            self.hn_points = 150
            self.hn_comments = 20
            self.is_trending = True
            self.is_viral = False
            self.hn_url = "http://hn.com"

    # Mock the return value
    mock_job = MagicMock()
    mock_job.result.return_value = [MockRow()]
    mock_bq_client.query.return_value = mock_job

    response = client.get("/api/feeds/daily_reads")
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 1
    article = data["articles"][0]
    
    # Check nested structures
    assert article["scores"]["final"] == 88
    assert article["content"]["content_type"] == "tutorial"
    assert article["social_proof"]["hn_points"] == 150
    assert article["social_proof"]["is_trending"] is True
