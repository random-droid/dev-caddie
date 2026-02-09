# cloudrun/services/rate_limiter.py

import os
import logging
from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import datetime
from typing import Dict, Any

from google.cloud import firestore

logger = logging.getLogger(__name__)

# Initialize Firestore client (uses default credentials on Cloud Run)
db = firestore.Client()

limiter = Limiter(key_func=get_remote_address)


class BudgetGuard:
    """
    Protect against cost-based attacks using Firestore for distributed state.
    Tracks API usage and enforces daily budget across all Cloud Run instances.
    """

    DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "2.00"))
    COST_PER_REQUEST_USD = 0.002  # Average Gemini cost
    MAX_REQUESTS_PER_IP = int(os.getenv("MAX_REQUESTS_PER_IP", "10"))

    # Firestore collection names
    COLLECTION_GLOBAL = "rate_limits_global"
    COLLECTION_PER_IP = "rate_limits_per_ip"

    @classmethod
    def _get_today(cls) -> str:
        """Get today's date as ISO string."""
        return datetime.now().date().isoformat()

    @classmethod
    def check_budget(cls, ip_address: str) -> bool:
        """
        Check BOTH Global Budget (Wallet) AND Per-IP Limits (Spam).
        Uses Firestore for distributed state across instances.
        """
        today = cls._get_today()

        try:
            # A. Global Guard - check daily system budget
            global_doc_ref = db.collection(cls.COLLECTION_GLOBAL).document(today)
            global_doc = global_doc_ref.get()

            if global_doc.exists:
                global_cost = global_doc.to_dict().get("cost_usd", 0.0)
                if global_cost >= cls.DAILY_BUDGET_USD:
                    raise HTTPException(
                        status_code=429,
                        detail="System usage limit reached. Try again tomorrow."
                    )

            # B. Per-IP Guard - check individual IP limits
            ip_doc_id = f"{today}_{ip_address.replace('.', '_').replace(':', '_')}"
            ip_doc_ref = db.collection(cls.COLLECTION_PER_IP).document(ip_doc_id)
            ip_doc = ip_doc_ref.get()

            if ip_doc.exists:
                ip_data = ip_doc.to_dict()
                if ip_data.get("count", 0) >= cls.MAX_REQUESTS_PER_IP:
                    raise HTTPException(
                        status_code=429,
                        detail="Your daily AI limit exceeded."
                    )
                if ip_data.get("suspicious_count", 0) > 5:
                    raise HTTPException(
                        status_code=403,
                        detail="Access blocked."
                    )

            return True

        except HTTPException:
            raise
        except Exception as e:
            # Log but don't block on Firestore errors - fail open with warning
            logger.warning(f"Firestore read error in check_budget: {e}")
            return True

    @classmethod
    def record_request(cls, ip_address: str, cost: float, is_suspicious: bool = False):
        """Record usage to BOTH Global and Per-IP counters in Firestore."""
        today = cls._get_today()

        try:
            # Update Global cost atomically
            global_doc_ref = db.collection(cls.COLLECTION_GLOBAL).document(today)
            global_doc_ref.set({
                "cost_usd": firestore.Increment(cost),
                "request_count": firestore.Increment(1),
                "last_updated": firestore.SERVER_TIMESTAMP
            }, merge=True)

            # Update Per-IP counters atomically
            ip_doc_id = f"{today}_{ip_address.replace('.', '_').replace(':', '_')}"
            ip_doc_ref = db.collection(cls.COLLECTION_PER_IP).document(ip_doc_id)

            update_data = {
                "count": firestore.Increment(1),
                "cost_usd": firestore.Increment(cost),
                "ip_address": ip_address,
                "date": today,
                "last_updated": firestore.SERVER_TIMESTAMP
            }

            if is_suspicious:
                update_data["suspicious_count"] = firestore.Increment(1)

            ip_doc_ref.set(update_data, merge=True)

        except Exception as e:
            # Log but don't fail the request on Firestore write errors
            logger.error(f"Firestore write error in record_request: {e}")

    @classmethod
    def get_usage_stats(cls, ip_address: str) -> Dict[str, Any]:
        """Get current usage stats from Firestore."""
        today = cls._get_today()

        stats = {
            "global_cost_usd": 0.0,
            "global_remaining_usd": cls.DAILY_BUDGET_USD,
            "per_ip_used": 0,
            "per_ip_remaining": cls.MAX_REQUESTS_PER_IP
        }

        try:
            # Get global stats
            global_doc = db.collection(cls.COLLECTION_GLOBAL).document(today).get()
            if global_doc.exists:
                global_cost = global_doc.to_dict().get("cost_usd", 0.0)
                stats["global_cost_usd"] = round(global_cost, 4)
                stats["global_remaining_usd"] = round(max(0, cls.DAILY_BUDGET_USD - global_cost), 4)

            # Get per-IP stats
            ip_doc_id = f"{today}_{ip_address.replace('.', '_').replace(':', '_')}"
            ip_doc = db.collection(cls.COLLECTION_PER_IP).document(ip_doc_id).get()
            if ip_doc.exists:
                ip_data = ip_doc.to_dict()
                stats["per_ip_used"] = ip_data.get("count", 0)
                stats["per_ip_remaining"] = max(0, cls.MAX_REQUESTS_PER_IP - stats["per_ip_used"])

        except Exception as e:
            logger.warning(f"Firestore read error in get_usage_stats: {e}")

        return stats
