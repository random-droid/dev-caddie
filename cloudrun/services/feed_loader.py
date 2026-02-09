"""
Load feeds from OPML into BigQuery.
Relocated from scripts/load_feeds_from_opml.py to be accessible within Cloud Run service context.
"""

import logging
from datetime import datetime
from google.cloud import bigquery
from services.opml_parser import OPMLParser, generate_feed_id

logger = logging.getLogger(__name__)

def load_feeds_to_bigquery(
    opml_path: str,
    project_id: str,
    dataset_id: str = "content_intelligence",
    table_id: str = "feeds_metadata"
):
    """
    Load feeds from OPML to BigQuery.
    
    Args:
        opml_path: Path to OPML file (relative to service root)
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
    """
    try:
        client = bigquery.Client(project=project_id)
        
        # Note: OPMLParser import is now from services.opml_parser
        parser = OPMLParser(opml_path)
        
        # Parse feeds
        feeds = parser.parse()
        logger.info(f"Parsed {len(feeds)} feeds from OPML")
        
        if not feeds:
            logger.warning("No feeds found in OPML")
            return

        # Prepare rows for BigQuery
        rows = []
        for feed in feeds:
            feed_id = generate_feed_id(feed.xml_url)
            rows.append({
                'feed_id': feed_id,
                'title': feed.title,
                'xml_url': feed.xml_url,
                'html_url': feed.html_url,
                'category': feed.category,
                'description': feed.description,
                'is_active': True,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            })
        
        # Insert or update (MERGE) via Temp Table
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        temp_table_id = f"{dataset_id}.temp_feeds_load_{int(datetime.utcnow().timestamp())}"
        temp_table_ref = f"{project_id}.{temp_table_id}"
        
        # 1. Load data to temp table
        logger.info(f"Loading {len(rows)} feeds to temp table {temp_table_ref}...")
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition="WRITE_TRUNCATE"
        )
        load_job = client.load_table_from_json(rows, temp_table_ref, job_config=job_config)
        load_job.result()
        
        # 2. Perform MERGE
        merge_query = f"""
        MERGE `{table_ref}` AS target
        USING `{temp_table_ref}` AS source
        ON target.feed_id = source.feed_id
        WHEN MATCHED THEN
            UPDATE SET
                title = source.title,
                xml_url = source.xml_url,
                html_url = source.html_url,
                category = source.category,
                description = source.description,
                is_active = source.is_active,
                updated_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
            INSERT (
                feed_id, title, xml_url, html_url, category, 
                description, is_active, created_at, updated_at
            )
            VALUES (
                source.feed_id, source.title, source.xml_url, 
                source.html_url, source.category, source.description, 
                source.is_active, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
            )
        """
        
        logger.info("Executing MERGE...")
        query_job = client.query(merge_query)
        query_job.result()
        
        # 3. Cleanup
        client.delete_table(temp_table_ref, not_found_ok=True)
        
        logger.info(f"✅ Successfully synced {len(rows)} feeds to {table_ref}")
        
    except Exception as e:
        logger.error(f"Failed to load feeds to BQ: {e}")
        raise
