# cloudrun/config.py

import os
from datetime import datetime

class Config:
    """Application configuration with vacation mode support."""
    
    # Vacation mode flag (set via environment variable)
    VACATION_MODE = os.getenv('VACATION_MODE', 'false').lower() == 'true'

    
    # When vacation mode ends
    VACATION_END_DATE = os.getenv('VACATION_END_DATE', None)
    
    # Lobsters Safety Settings
    LOBSTERS_USER_AGENT = "ContentIntelligenceHub/1.0 (+mailto:your-email@example.com)"
    LOBSTERS_MAX_RPS = 1.0

    @classmethod
    def is_vacation_mode(cls) -> bool:
        """Check if currently in vacation mode."""
        if not cls.VACATION_MODE:
            return False
        
        # Auto-disable if vacation period ended
        if cls.VACATION_END_DATE:
            try:
                end_date = datetime.fromisoformat(cls.VACATION_END_DATE)
                if datetime.now() > end_date:
                    return False
            except (ValueError, TypeError, AttributeError):
                # Invalid date format - continue with vacation mode enabled
                pass
        
        return True
