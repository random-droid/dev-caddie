# cloudrun/services/input_validator.py

import re
from typing import Tuple, List
from fastapi import HTTPException

class InputValidator:
    """
    First line of defense against prompt injection.
    Validates and sanitizes all user inputs.
    """
    
    # Suspicious patterns that indicate injection attempts
    INJECTION_PATTERNS = [
        # System instruction overrides
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'ignore\s+(all\s+)?prior\s+instructions?',
        r'disregard\s+(all\s+)?previous\s+instructions?',
        r'forget\s+(all\s+)?previous\s+instructions?',
        
        # Role-playing attacks
        r'you\s+are\s+now\s+(?:a\s+)?(?:dan|do\s+anything\s+now)',
        r'pretend\s+you\s+are',
        r'act\s+as\s+(?:if\s+)?you\s+(?:are|were)',
        r'roleplay\s+as',
        
        # System/admin claims
        r'(?:i\s+am|i\'m)\s+(?:the\s+)?(?:admin|administrator|system|developer)',
        r'system\s*[:]\s*',
        r'admin\s*[:]\s*',
        
        # Prompt extraction attempts
        r'show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)',
        r'what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions)',
        r'repeat\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)',
        
        # Jailbreak patterns
        r'developer\s+mode',
        r'debug\s+mode',
        r'unlimited\s+mode',
        r'jailbreak',
        
        # Encoding/obfuscation attempts
        r'base64',
        r'rot13',
        r'\\x[0-9a-f]{2}',  # hex encoding
        
        # Multiple language mixing (common in sophisticated attacks)
        r'[\u4e00-\u9fff].*ignore',  # Chinese + English
        r'[\u0400-\u04ff].*system',   # Cyrillic + English
    ]
    
    # Compile patterns for performance
    COMPILED_PATTERNS = [
        re.compile(pattern, re.IGNORECASE | re.DOTALL) 
        for pattern in INJECTION_PATTERNS
    ]
    
    # Maximum input lengths
    MAX_QUERY_LENGTH = 500      # characters
    MAX_CODE_LENGTH = 10000     # characters for code review
    MAX_URL_LENGTH = 2000       # characters
    
    # Rate limiting per pattern detection
    SUSPICIOUS_SCORE_THRESHOLD = 3
    
    @classmethod
    def validate_user_query(cls, query: str) -> Tuple[bool, str, int]:
        """
        Validate user query for Content Intelligence Assistant.
        
        Returns:
            (is_valid, sanitized_query, suspicion_score)
        """
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Length check
        if len(query) > cls.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Query too long. Max {cls.MAX_QUERY_LENGTH} characters."
            )
        
        # Check for injection patterns
        suspicion_score = 0
        detected_patterns = []
        
        for pattern in cls.COMPILED_PATTERNS:
            if pattern.search(query):
                suspicion_score += 1
                detected_patterns.append(pattern.pattern)
        
        # Block if highly suspicious
        if suspicion_score >= cls.SUSPICIOUS_SCORE_THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail="Input rejected: suspicious patterns detected"
            )
        
        # Sanitize
        sanitized = cls._sanitize_input(query)
        
        return True, sanitized, suspicion_score
    
    @classmethod
    def validate_code_snippet(cls, code: str) -> Tuple[bool, str, int]:
        """
        Validate code snippet for LLM Review.
        
        Returns:
            (is_valid, sanitized_code, suspicion_score)
        """
        if not code or not code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        
        # Length check
        if len(code) > cls.MAX_CODE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Code too long. Max {cls.MAX_CODE_LENGTH} characters."
            )
        
        # Check for injection in comments (common hiding place)
        suspicion_score = 0
        
        # Extract comments
        comment_patterns = [
            r'#.*$',           # Python comments
            r'//.*$',          # C++/Java single-line
            r'/\*.*?\*/',      # C++/Java multi-line
        ]
        
        for comment_pattern in comment_patterns:
            comments = re.findall(comment_pattern, code, re.MULTILINE | re.DOTALL)
            for comment in comments:
                for injection_pattern in cls.COMPILED_PATTERNS:
                    if injection_pattern.search(comment):
                        suspicion_score += 1
        
        # Check for embedded prompts in strings
        string_patterns = [
            r'".*?"',          # Double quotes
            r"'.*?'",          # Single quotes
            r'""".*?"""',      # Python docstrings
        ]
        
        for string_pattern in string_patterns:
            strings = re.findall(string_pattern, code, re.DOTALL)
            for string in strings:
                for injection_pattern in cls.COMPILED_PATTERNS:
                    if injection_pattern.search(string):
                        suspicion_score += 1
        
        if suspicion_score >= cls.SUSPICIOUS_SCORE_THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail="Code rejected: suspicious patterns in comments/strings"
            )
        
        # Sanitize
        sanitized = cls._sanitize_input(code)
        
        return True, sanitized, suspicion_score
    
    @staticmethod
    def _sanitize_input(text: str) -> str:
        """
        Sanitize input by:
        1. Removing null bytes
        2. Normalizing whitespace
        3. Removing control characters
        4. Limiting consecutive special characters
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove other control characters (except newline, tab, carriage return)
        text = ''.join(
            char for char in text 
            if ord(char) >= 32 or char in '\n\r\t'
        )
        
        # Normalize excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Limit consecutive special characters (potential obfuscation)
        text = re.sub(r'([!@#$%^&*()_+=\[\]{}|\\:;"\'<>,.?/~`-])\1{5,}', r'\1\1\1', text)
        
        return text.strip()
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL for article analysis."""
        if len(url) > InputValidator.MAX_URL_LENGTH:
            raise HTTPException(status_code=400, detail="URL too long")
        
        # Must be HTTP/HTTPS
        if not re.match(r'^https?://', url, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Invalid URL scheme")
        
        # Block local/private IPs (SSRF prevention)
        blocked_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'0\.0\.0\.0',
            r'10\.\d+\.\d+\.\d+',      # Private IP
            r'172\.(1[6-9]|2\d|3[01])', # Private IP
            r'192\.168\.',              # Private IP
            r'\[::\]',                  # IPv6 localhost
        ]
        
        for pattern in blocked_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                raise HTTPException(status_code=400, detail="Invalid URL")
        
        return True
    