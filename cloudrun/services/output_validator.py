# cloudrun/services/output_validator.py

import re
from typing import Dict, Any, Tuple

class OutputValidator:
    """
    Validate LLM outputs before returning to user.
    Last line of defense against leaked prompts or inappropriate content.
    """
    
    # Patterns that indicate prompt leakage
    LEAKED_PROMPT_PATTERNS = [
        r'STRICT RULES:',
        r'You are a (?:helpful|code review)',
        r'IGNORE any instructions',
        r'====== START OF',
        r'USER INTERESTS:',
        r'AVAILABLE ARTICLES:',
    ]
    
    # Patterns indicating the LLM was compromised
    COMPROMISE_INDICATORS = [
        r'I am now (?:a|acting as)',
        r'developer mode activated',
        r'jailbreak successful',
        r'restrictions? removed',
    ]
    
    
    # PII Patterns
    PII_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',                        # Phone (simple US)
        r'\b\d{3}-\d{2}-\d{4}\b',                                # SSN
    ]

    @classmethod
    def contains_pii(cls, text: str) -> bool:
        """
        Check if text contains PII.
        
        Args:
            text: String to check (e.g. JSON dump of intent)
            
        Returns:
            True if PII detected
        """
        for pattern in cls.PII_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    @classmethod
    def validate_assistant_response(cls, response: str) -> Tuple[bool, str]:
        """
        Validate assistant response before returning to user.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for leaked system prompt
        for pattern in cls.LEAKED_PROMPT_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return False, "Response contained sensitive information"
        
        # Check for compromise indicators
        for pattern in cls.COMPROMISE_INDICATORS:
            if re.search(pattern, response, re.IGNORECASE):
                return False, "Response indicated security compromise"
        
        # Check for PII
        if cls.contains_pii(response):
            return False, "Response contained PII"
            
        # Length check (prevent extremely long responses = potential attack)
        if len(response) > 10000:  # 10K characters
            return False, "Response too long"
        
        return True, ""
    
    @classmethod
    def sanitize_response(cls, response: str) -> str:
        """
        Remove any accidental system prompt leakage.
        """
        # Remove delimiter artifacts
        response = re.sub(
            r'====== (?:START|END) OF .*? ======\n?',
            '',
            response,
            flags=re.IGNORECASE
        )
        
        # Remove internal thought processes (if any)
        response = re.sub(
            r'\[INTERNAL:.*?\]',
            '',
            response,
            flags=re.DOTALL
        )
        
        return response.strip()
    