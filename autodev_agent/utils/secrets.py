"""
Utility modules for AutoDevAgent.
Includes secret masking, metrics, and other helpers.
"""

import re
import os
from typing import Optional


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret value, showing only first few characters."""
    if not value or len(value) <= visible_chars:
        return "*" * len(value) if value else ""
    return value[:visible_chars] + "*" * (len(value) - visible_chars)


def mask_api_key(api_key: str) -> str:
    """Mask an API key for logging."""
    return mask_secret(api_key, 8)


class SecretMasker:
    """Masks secrets in log messages."""
    
    PATTERNS = [
        (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]+)', re.IGNORECASE), 'api_key="***"'),
        (re.compile(r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]+)', re.IGNORECASE), 'token="***"'),
        (re.compile(r'secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]+)', re.IGNORECASE), 'secret="***"'),
        (re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\',\s]+)', re.IGNORECASE), 'password="***"'),
        (re.compile(r'Bearer\s+([a-zA-Z0-9_\-\.]+)'), 'Bearer ***'),
        (re.compile(r'sk-[a-zA-Z0-9]{32,}'), 'sk-***'),  # OpenAI keys
    ]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def mask(self, text: str) -> str:
        """Mask all secrets in text."""
        if not self.enabled or not text:
            return text
        
        result = text
        for pattern, replacement in self.PATTERNS:
            result = pattern.sub(replacement, result)
        
        return result


def get_env_or_default(name: str, default: str) -> str:
    """Get environment variable or return default."""
    return os.environ.get(name, default)


def is_truthy(value: str) -> bool:
    """Check if a string value is truthy."""
    return value.lower() in ('1', 'true', 'yes', 'on')


__all__ = [
    "mask_secret",
    "mask_api_key",
    "SecretMasker",
    "get_env_or_default",
    "is_truthy",
]
