"""Tests for cache module."""

import pytest
import time
from pathlib import Path
import tempfile
import shutil

from autodev_agent.cache.simple import SimpleCache, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_creation(self):
        """Test creating a cache entry."""
        from datetime import datetime
        
        entry = CacheEntry(
            key="test123",
            prompt="What is Python?",
            response="Python is a programming language",
            model="gpt-4",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            created_at=datetime.now().isoformat(),
            ttl_seconds=3600
        )
        
        assert entry.key == "test123"
        assert entry.input_tokens == 10
        assert entry.output_tokens == 20
        assert entry.cost_usd == 0.001
    
    def test_is_expired(self):
        """Test expiration check."""
        from datetime import datetime, timedelta
        
        # Not expired (TTL in future)
        entry1 = CacheEntry(
            key="test1",
            prompt="test",
            response="test",
            model="test",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            created_at=datetime.now().isoformat(),
            ttl_seconds=3600
        )
        assert entry1.is_expired() is False
        
        # Expired (TTL in past)
        entry2 = CacheEntry(
            key="test2",
            prompt="test",
            response="test",
            model="test",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            created_at=(datetime.now() - timedelta(hours=2)).isoformat(),
            ttl_seconds=3600
        )
        assert entry2.is_expired() is True
        
        # No expiry (ttl_seconds <= 0)
        entry3 = CacheEntry(
            key="test3",
            prompt="test",
            response="test",
            model="test",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            created_at=(datetime.now() - timedelta(days=1)).isoformat(),
            ttl_seconds=0
        )
        assert entry3.is_expired() is False
    
    def test_to_dict_and_back(self):
        """Test serialization and deserialization."""
        from datetime import datetime
        
        original = CacheEntry(
            key="test123",
            prompt="Test prompt",
            response="Test response",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
            created_at=datetime.now().isoformat(),
            ttl_seconds=7200
        )
        
        data = original.to_dict()
        restored = CacheEntry.from_dict(data)
        
        assert restored.key == original.key
        assert restored.prompt == original.prompt
        assert restored.response == original.response
        assert restored.model == original.model
        assert restored.input_tokens == original.input_tokens


class TestSimpleCache:
    """Tests for SimpleCache class."""
    
    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def cache(self, cache_dir):
        """Create cache instance."""
        return SimpleCache(cache_dir=cache_dir, ttl_seconds=3600)
    
    def test_initialization(self, cache, cache_dir):
        """Test cache initialization."""
        assert cache.cache_dir == Path(cache_dir)
        assert cache.ttl_seconds == 3600
        
        # Check index file was created
        index_file = Path(cache_dir) / "index.json"
        assert index_file.exists()
    
    def test_get_miss(self, cache):
        """Test cache miss for non-existent key."""
        result = cache.get(
            prompt="Never cached this",
            model="gpt-4",
            temperature=0.7
        )
        assert result is None
    
    def test_set_and_get(self, cache):
        """Test setting and getting a cache entry."""
        # Set cache
        key = cache.set(
            prompt="What is Python?",
            response="Python is a programming language",
            model="gpt-4",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            temperature=0.7
        )
        
        assert key is not None
        
        # Get cache
        result = cache.get(
            prompt="What is Python?",
            model="gpt-4",
            temperature=0.7
        )
        
        assert result is not None
        assert result['response'] == "Python is a programming language"
        assert result['model'] == "gpt-4"
        assert result['input_tokens'] == 10
        assert result['cached'] is True
    
    def test_cache_key_includes_temperature(self, cache):
        """Test that different temperatures create different cache keys."""
        # Set with temperature 0.5
        cache.set(
            prompt="Same prompt",
            response="Response A",
            model="gpt-4",
            input_tokens=5,
            output_tokens=10,
            temperature=0.5
        )
        
        # Set with temperature 0.9
        cache.set(
            prompt="Same prompt",
            response="Response B",
            model="gpt-4",
            input_tokens=5,
            output_tokens=10,
            temperature=0.9
        )
        
        # Get with temperature 0.5
        result_a = cache.get("Same prompt", "gpt-4", temperature=0.5)
        assert result_a['response'] == "Response A"
        
        # Get with temperature 0.9
        result_b = cache.get("Same prompt", "gpt-4", temperature=0.9)
        assert result_b['response'] == "Response B"
    
    def test_delete(self, cache):
        """Test deleting a cache entry."""
        # Set cache
        cache.set(
            prompt="To be deleted",
            response="Gone soon",
            model="gpt-4",
            input_tokens=5,
            output_tokens=5
        )
        
        # Verify it exists
        result = cache.get("To be deleted", "gpt-4")
        assert result is not None
        
        # Delete
        success = cache.delete(cache._get_cache_key("To be deleted", "gpt-4", 0.7, None))
        assert success is True
        
        # Verify it's gone
        result = cache.get("To be deleted", "gpt-4")
        assert result is None
    
    def test_clear(self, cache):
        """Test clearing all cache entries."""
        # Add multiple entries
        cache.set("Prompt 1", "Response 1", "gpt-4", 5, 5)
        cache.set("Prompt 2", "Response 2", "gpt-4", 5, 5)
        cache.set("Prompt 3", "Response 3", "gpt-4", 5, 5)
        
        # Clear
        cache.clear()
        
        # Verify all are gone
        stats = cache.stats()
        assert stats['entries'] == 0
    
    def test_stats(self, cache):
        """Test getting cache statistics."""
        # Add some entries
        cache.set("Prompt 1", "Response 1", "gpt-4", 5, 5)
        cache.set("Prompt 2", "Response 2", "gpt-4", 10, 10)
        
        stats = cache.stats()
        
        assert stats['entries'] == 2
        assert stats['size_mb'] >= 0
        assert stats['max_size_mb'] == 1000
        assert 'directory' in stats
    
    def test_list_entries(self, cache):
        """Test listing cache entries."""
        # Add entries
        cache.set("Prompt 1", "Response 1", "gpt-4", 5, 5)
        cache.set("Prompt 2", "Response 2", "llama3", 10, 10)
        
        entries = cache.list_entries()
        
        assert len(entries) == 2
        
        # Check fields
        for entry in entries:
            assert 'key' in entry
            assert 'model' in entry
            assert 'created_at' in entry
            assert 'tokens' in entry
            assert 'expired' in entry
    
    def test_ttl_expiration(self, cache_dir):
        """Test TTL-based expiration."""
        # Create cache with very short TTL
        cache = SimpleCache(cache_dir=cache_dir, ttl_seconds=1)
        
        # Set cache
        cache.set(
            prompt="Short lived",
            response="Gone soon",
            model="gpt-4",
            input_tokens=5,
            output_tokens=5
        )
        
        # Should exist immediately
        result = cache.get("Short lived", "gpt-4")
        assert result is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired now
        result = cache.get("Short lived", "gpt-4")
        assert result is None
