# cloudrun/services/vacation_limiter.py

from datetime import datetime, date
from fastapi import HTTPException
from google.cloud import firestore
import logging

logger = logging.getLogger(__name__)

# Initialize Firestore client
db = firestore.Client()

class VacationLimiter:
    """Ultra-conservative rate limiting during vacation mode using Firestore."""

    # Much stricter limits when you're away
    VACATION_LIMITS = {
        'requests_per_day': 10,      # Global daily limit
        'requests_per_ip_per_day': 2 # Per-IP daily limit
    }

    # Firestore collection
    COLLECTION = "vacation_limits"

    @classmethod
    def check_vacation_limits(cls, ip_address: str) -> bool:
        """Enforce ultra-strict limits during vacation using Firestore."""
        today = date.today().isoformat()

        try:
            # Check global daily limit
            global_doc = db.collection(cls.COLLECTION).document(f"global_{today}").get()
            global_count = global_doc.to_dict().get('count', 0) if global_doc.exists else 0

            if global_count >= cls.VACATION_LIMITS['requests_per_day']:
                raise HTTPException(
                    status_code=503,
                    detail="Daily request quota reached. Service in limited availability mode."
                )

            # Check per-IP limit
            ip_safe = ip_address.replace('.', '_').replace(':', '_')
            ip_doc = db.collection(cls.COLLECTION).document(f"ip_{ip_safe}_{today}").get()
            ip_count = ip_doc.to_dict().get('count', 0) if ip_doc.exists else 0

            if ip_count >= cls.VACATION_LIMITS['requests_per_ip_per_day']:
                raise HTTPException(
                    status_code=429,
                    detail="You've reached your request limit for today. Please try tomorrow."
                )

            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Firestore error in vacation limiter check: {e}")
            # Fail open - allow request if Firestore is down
            return True

    @classmethod
    def record_vacation_request(cls, ip_address: str):
        """Record request to Firestore during vacation mode."""
        today = date.today().isoformat()

        try:
            # Increment global counter
            global_ref = db.collection(cls.COLLECTION).document(f"global_{today}")
            global_ref.set({
                'count': firestore.Increment(1),
                'last_updated': firestore.SERVER_TIMESTAMP
            }, merge=True)

            # Increment per-IP counter
            ip_safe = ip_address.replace('.', '_').replace(':', '_')
            ip_ref = db.collection(cls.COLLECTION).document(f"ip_{ip_safe}_{today}")
            ip_ref.set({
                'count': firestore.Increment(1),
                'ip_address': ip_address,
                'last_updated': firestore.SERVER_TIMESTAMP
            }, merge=True)

        except Exception as e:
            logger.error(f"Firestore error recording vacation request: {e}")
            # Don't fail the request on write errors
