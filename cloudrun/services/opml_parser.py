"""
OPML Parser for Content Intelligence Platform

Parses feeds.opml and extracts feed metadata for pipeline.
"""

import xml.etree.ElementTree as ET
import hashlib
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path


def generate_feed_id(xml_url: str) -> str:
    """Generate unique feed ID from URL."""
    return hashlib.md5(xml_url.encode()).hexdigest()


@dataclass
class RSSFeed:
    """RSS Feed metadata from OPML."""
    title: str
    xml_url: str
    html_url: str
    category: str
    description: str = ""


class OPMLParser:
    """Parse OPML file and extract feed information."""
    
    def __init__(self, opml_path: str):
        """Initialize parser with OPML file path."""
        self.opml_path = Path(opml_path)
        if not self.opml_path.exists():
            raise FileNotFoundError(f"OPML file not found: {opml_path}")
    
    def parse(self) -> List[RSSFeed]:
        """
        Parse OPML and return list of RSS feeds.
        
        Returns:
            List of RSSFeed objects with metadata
        """
        tree = ET.parse(self.opml_path)
        root = tree.getroot()
        
        feeds = []
        
        # Navigate to body element
        body = root.find('.//body')
        if body is None:
            raise ValueError("Invalid OPML: no body element found")
        
        # Process all outline elements recursively to handle nested structures
        self._process_outlines(body, feeds, 'Uncategorized')

        return feeds

    def _process_outlines(self, parent: ET.Element, feeds: List[RSSFeed], category: str):
        """Recursively process outline elements to find feeds."""
        for outline in parent.findall('./outline'):
            if outline.get('type') == 'rss':
                # This is a feed
                feed = self._parse_feed_outline(outline, category)
                if feed:
                    feeds.append(feed)
            else:
                # This is a category/folder - recurse into it
                new_category = outline.get('text', category)
                self._process_outlines(outline, feeds, new_category)
    
    def _parse_feed_outline(self, outline: ET.Element, category: str) -> RSSFeed:
        """Parse a single feed outline element."""
        title = outline.get('title') or outline.get('text', 'Unknown Feed')
        xml_url = outline.get('xmlUrl', '')
        html_url = outline.get('htmlUrl', '')
        description = outline.get('description', '')
        
        if not xml_url:
            return None
        
        return RSSFeed(
            title=title,
            xml_url=xml_url,
            html_url=html_url,
            category=category,
            description=description
        )
    
    def get_feeds_by_category(self) -> Dict[str, List[RSSFeed]]:
        """
        Get feeds grouped by category.
        
        Returns:
            Dictionary mapping category names to lists of feeds
        """
        feeds = self.parse()
        
        by_category = {}
        for feed in feeds:
            if feed.category not in by_category:
                by_category[feed.category] = []
            by_category[feed.category].append(feed)
        
        return by_category
    
    def get_feed_count(self) -> int:
        """Get total number of feeds in OPML."""
        return len(self.parse())
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        feeds = self.parse()
        return sorted(list(set(feed.category for feed in feeds)))


def export_to_json(opml_path: str, output_path: str):
    """
    Export OPML feeds to JSON format.
    
    Args:
        opml_path: Path to OPML file
        output_path: Path to output JSON file
    """
    import json
    
    parser = OPMLParser(opml_path)
    feeds = parser.parse()
    
    feed_data = [
        {
            'title': feed.title,
            'xml_url': feed.xml_url,
            'html_url': feed.html_url,
            'category': feed.category,
            'description': feed.description
        }
        for feed in feeds
    ]
    
    with open(output_path, 'w') as f:
        json.dump(feed_data, f, indent=2)
    
    print(f"Exported {len(feed_data)} feeds to {output_path}")


if __name__ == "__main__":
    # Example usage
    parser = OPMLParser("feeds.opml")
    
    # Get all feeds
    feeds = parser.parse()
    print(f"Total feeds: {len(feeds)}")
    
    # Get feeds by category
    by_category = parser.get_feeds_by_category()
    for category, category_feeds in by_category.items():
        print(f"\n{category}: {len(category_feeds)} feeds")
        for feed in category_feeds[:3]:  # Show first 3
            print(f"  - {feed.title}")
    
    # Export to JSON
    export_to_json("feeds.opml", "feeds.json")
    