# cloudrun/services/security_monitor.py

import logging
from datetime import datetime
from google.cloud import monitoring_v3
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SecurityMonitor:
    """
    Monitor for security events and alert on suspicious activity.
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
    
    def log_injection_attempt(
        self,
        ip_address: str,
        user_input: str,
        detected_patterns: List[str],
        suspicion_score: int
    ):
        """
        Log prompt injection attempt with full context.
        """
        logger.warning(
            "SECURITY: Prompt injection attempt detected",
            extra={
                'ip_address': ip_address,
                'suspicion_score': suspicion_score,
                'detected_patterns': detected_patterns,
                'input_length': len(user_input),
                'input_preview': user_input[:200],  # First 200 chars only
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Send custom metric to Cloud Monitoring
        self._send_security_metric(
            'injection_attempts',
            suspicion_score,
            {'ip': ip_address}
        )
    
    def log_budget_exceeded(self, ip_address: str, usage: Dict[str, Any]):
        """Log when budget limits are hit."""
        logger.warning(
            "SECURITY: Budget limit exceeded",
            extra={
                'ip_address': ip_address,
                'usage': usage,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        self._send_security_metric(
            'budget_exceeded',
            1,
            {'ip': ip_address}
        )
    
    def log_output_validation_failure(self, response_preview: str, reason: str):
        """Log when output validation catches something."""
        logger.error(
            "SECURITY: Output validation failed",
            extra={
                'reason': reason,
                'response_preview': response_preview[:200],
                'timestamp': datetime.now().isoformat()
            }
        )
        
        self._send_security_metric('output_validation_failures', 1, {})
    
    def _send_security_metric(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str]
    ):
        """Send custom security metric to Cloud Monitoring."""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = f'custom.googleapis.com/security/{metric_name}'
            series.metric.labels.update(labels)
            
            now = datetime.now()
            seconds = int(now.timestamp())
            nanos = int((now.timestamp() - seconds) * 10**9)
            
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )
            
            point = monitoring_v3.Point(
                {"interval": interval, "value": {"double_value": value}}
            )
            
            series.points = [point]
            self.client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
        except Exception as e:
            logger.error(f"Failed to send security metric: {e}")