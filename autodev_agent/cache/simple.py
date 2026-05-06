"""
Simple cache module for AutoDevAgent.
Provides basic hash-based caching for LLM responses.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    prompt: str
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: str
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        
        created = datetime.fromisoformat(self.created_at)
        expiry = created + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'prompt': self.prompt,
            'response': self.response,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cost_usd': self.cost_usd,
            'created_at': self.created_at,
            'ttl_seconds': self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class SimpleCache:
    """
    Simple file-based cache for LLM responses.
    
    Uses MD5 hash of prompt + parameters as cache key.
    Stores entries as JSON files for easy inspection.
    """
    
    def __init__(self, cache_dir: str, ttl_seconds: int = 86400, max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        
        # In-memory index for faster lookups
        self._index: Dict[str, str] = {}  # key -> filename
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize cache directory and load index."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing index
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache index: {e}")
                self._index = {}
        else:
            # Create empty index file
            self._save_index()
        
        # Clean up expired entries on startup
        self.cleanup()
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float, system_prompt: Optional[str] = None) -> str:
        """Generate cache key from request parameters."""
        key_data = f"{prompt}:{model}:{temperature}:{system_prompt or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_entry_path(self, key: str) -> Path:
        """Get file path for a cache entry."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, prompt: str, model: str, temperature: float = 0.7, system_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a prompt.
        
        Returns None if not found or expired.
        """
        key = self._get_cache_key(prompt, model, temperature, system_prompt)
        
        # Check in-memory index first
        filename = self._index.get(key)
        if not filename:
            return None
        
        entry_path = self.cache_dir / filename
        if not entry_path.exists():
            # File was deleted, remove from index
            del self._index[key]
            return None
        
        try:
            with open(entry_path, 'r') as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            
            # Check expiration
            if entry.is_expired():
                self.delete(key)
                return None
            
            return {
                'response': entry.response,
                'model': entry.model,
                'input_tokens': entry.input_tokens,
                'output_tokens': entry.output_tokens,
                'cost_usd': entry.cost_usd,
                'cached': True
            }
            
        except Exception as e:
            print(f"Warning: Error reading cache entry {key}: {e}")
            return None
    
    def set(
        self,
        prompt: str,
        response: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Cache a response.
        
        Returns the cache key.
        """
        key = self._get_cache_key(prompt, model, temperature, system_prompt)
        
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            created_at=datetime.now().isoformat(),
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        )
        
        # Save to file
        entry_path = self._get_entry_path(key)
        with open(entry_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        # Update index
        self._index[key] = entry_path.name
        self._save_index()
        
        # Check size limit
        self._enforce_size_limit()
        
        return key
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        if key not in self._index:
            return False
        
        filename = self._index[key]
        entry_path = self.cache_dir / filename
        
        try:
            if entry_path.exists():
                entry_path.unlink()
            
            del self._index[key]
            self._save_index()
            return True
            
        except Exception as e:
            print(f"Error deleting cache entry {key}: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries."""
        # Delete all .json files except index
        for file in self.cache_dir.glob("*.json"):
            if file.name != "index.json":
                file.unlink()
        
        self._index = {}
        self._save_index()
    
    def cleanup(self):
        """Remove expired entries."""
        expired_keys = []
        
        for key, filename in list(self._index.items()):
            entry_path = self.cache_dir / filename
            
            if not entry_path.exists():
                expired_keys.append(key)
                continue
            
            try:
                with open(entry_path, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry.from_dict(data)
                if entry.is_expired():
                    expired_keys.append(key)
            except Exception:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            print(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    
    def _save_index(self):
        """Save index to disk."""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache index: {e}")
    
    def _enforce_size_limit(self):
        """Remove oldest entries if cache exceeds size limit."""
        try:
            total_size_mb = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.json")
                if f.name != "index.json"
            ) / (1024 * 1024)
            
            if total_size_mb > self.max_size_mb:
                # Get entries sorted by creation time
                entries = []
                for key, filename in self._index.items():
                    entry_path = self.cache_dir / filename
                    if entry_path.exists():
                        try:
                            with open(entry_path, 'r') as f:
                                data = json.load(f)
                            entries.append((key, data.get('created_at', ''), entry_path))
                        except Exception:
                            continue
                
                # Sort by creation time (oldest first)
                entries.sort(key=lambda x: x[1])
                
                # Remove oldest until under limit
                for key, _, path in entries:
                    if total_size_mb <= self.max_size_mb * 0.9:  # Leave some buffer
                        break
                    
                    file_size = path.stat().st_size / (1024 * 1024)
                    path.unlink()
                    del self._index[key]
                    total_size_mb -= file_size
                
                self._save_index()
                
        except Exception as e:
            print(f"Warning: Error enforcing cache size limit: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_mb = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.json")
            if f.name != "index.json"
        ) / (1024 * 1024)
        
        return {
            'entries': len(self._index),
            'size_mb': round(total_size_mb, 2),
            'max_size_mb': self.max_size_mb,
            'directory': str(self.cache_dir)
        }
    
    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        entries = []
        
        for key, filename in self._index.items():
            entry_path = self.cache_dir / filename
            
            if not entry_path.exists():
                continue
            
            try:
                with open(entry_path, 'r') as f:
                    data = json.load(f)
                
                entries.append({
                    'key': key[:16] + '...',  # Truncate for readability
                    'model': data.get('model', 'unknown'),
                    'created_at': data.get('created_at', 'unknown'),
                    'tokens': data.get('input_tokens', 0) + data.get('output_tokens', 0),
                    'expired': CacheEntry.from_dict(data).is_expired()
                })
            except Exception:
                continue
        
        return sorted(entries, key=lambda x: x.get('created_at', ''), reverse=True)
