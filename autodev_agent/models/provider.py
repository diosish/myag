"""
Model Provider module for AutoDevAgent.
Supports multiple LLM providers (OpenAI, Ollama) with unified interface.
Implements prompt caching and token counting.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..models.config import Config, ModelConfig
from ..models.domain import TokenUsage


class ModelResponse:
    """Response from an LLM model."""
    
    def __init__(
        self,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        raw_response: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.model = model
        self.raw_response = raw_response or {}
        self.timestamp = datetime.now()
    
    @property
    def cost_usd(self) -> float:
        """Calculate cost based on token usage (needs model info)."""
        # This should be calculated by the provider with actual model rates
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'model': self.model,
            'timestamp': self.timestamp.isoformat()
        }


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, config: Config):
        self.config = config
        self._cache: Dict[str, ModelResponse] = {}
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass
    
    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        return self.config.get_model_info(model_name)
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str
    ) -> float:
        """Calculate cost for token usage."""
        model_info = self.get_model_info(model_name)
        if not model_info:
            return 0.0
        
        cost = (
            (input_tokens / 1000) * model_info.cost_per_1k_input +
            (output_tokens / 1000) * model_info.cost_per_1k_output
        )
        return cost
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (simple approximation)."""
        # Simple heuristic: ~4 characters per token for English
        # For production, use tiktoken or similar
        return len(text) // 4
    
    def _get_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_name: str,
        temperature: float
    ) -> str:
        """Generate cache key for a request."""
        key_data = f"{prompt}:{system_prompt}:{model_name}:{temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_name: str,
        temperature: float
    ) -> Optional[ModelResponse]:
        """Get response from cache."""
        if not self.config.cache.enabled:
            return None
        
        cache_key = self._get_cache_key(prompt, system_prompt, model_name, temperature)
        return self._cache.get(cache_key)
    
    def _save_to_cache(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_name: str,
        temperature: float,
        response: ModelResponse
    ):
        """Save response to cache."""
        if not self.config.cache.enabled:
            return
        
        cache_key = self._get_cache_key(prompt, system_prompt, model_name, temperature)
        self._cache[cache_key] = response


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            if not self.config.llm.openai or not self.config.llm.openai.api_key:
                print("Warning: OpenAI API key not configured")
                return
            
            self.client = OpenAI(
                api_key=self.config.llm.openai.api_key,
                base_url=self.config.llm.openai.base_url
            )
        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return self.client is not None and bool(self.config.llm.openai)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate response using OpenAI API."""
        
        if not self.is_available():
            raise RuntimeError("OpenAI provider is not available")
        
        # Check cache first
        model_name = model_name or self.config.llm.default_model
        cached = self._get_from_cache(prompt, system_prompt, model_name, temperature)
        if cached:
            print(f"[Cache Hit] Using cached response for model {model_name}")
            return cached
        
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Get max tokens from config if not specified
        if max_tokens is None:
            model_info = self.get_model_info(model_name)
            max_tokens = model_info.max_tokens if model_info else self.config.tokens.max_tokens_per_request
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens, model_name)
            
            result = ModelResponse(
                content=content or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                raw_response=response.dict()
            )
            
            # Add cost to response
            result._cost_usd = cost
            
            # Save to cache
            self._save_to_cache(prompt, system_prompt, model_name, temperature, result)
            
            print(f"[OpenAI] Model: {model_name}, Tokens: {result.total_tokens}, Cost: ${cost:.4f}")
            
            return result
            
        except Exception as e:
            print(f"[OpenAI Error] {str(e)}")
            raise


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.base_url = config.llm.ollama.base_url if config.llm.ollama else "http://localhost:11434"
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None:
            try:
                import requests
                self._client = requests
            except ImportError:
                print("Warning: requests package not installed")
                return None
        return self._client
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if not self.client:
            return False
        
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Ollama API."""
        
        if not self.is_available():
            raise RuntimeError("Ollama provider is not available")
        
        # Check cache first
        model_name = model_name or "llama3"
        cached = self._get_from_cache(prompt, system_prompt, model_name, temperature)
        if cached:
            print(f"[Cache Hit] Using cached response for model {model_name}")
            return cached
        
        # Build prompt with system message
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("response", "")
            
            # Ollama doesn't always provide token counts
            input_tokens = self.count_tokens(full_prompt)
            output_tokens = self.count_tokens(content)
            
            result = ModelResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
                raw_response=data
            )
            
            # Save to cache
            self._save_to_cache(prompt, system_prompt, model_name, temperature, result)
            
            print(f"[Ollama] Model: {model_name}, Tokens: {result.total_tokens}, Cost: $0.00 (local)")
            
            return result
            
        except Exception as e:
            print(f"[Ollama Error] {str(e)}")
            raise


class ModelManager:
    """Manages multiple model providers with fallback support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers: Dict[str, ModelProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        # Initialize OpenAI
        if self.config.llm.openai:
            openai_provider = OpenAIProvider(self.config)
            if openai_provider.is_available():
                self.providers['openai'] = openai_provider
                print("[ModelManager] OpenAI provider initialized")
            else:
                print("[ModelManager] OpenAI provider not available")
        
        # Initialize Ollama
        if self.config.llm.ollama:
            ollama_provider = OllamaProvider(self.config)
            if ollama_provider.is_available():
                self.providers['ollama'] = ollama_provider
                print("[ModelManager] Ollama provider initialized")
            else:
                print("[ModelManager] Ollama provider not available (make sure Ollama is running)")
    
    def get_provider(self, model_name: str) -> Optional[ModelProvider]:
        """Get appropriate provider for a model."""
        # Determine which provider handles this model
        if self.config.llm.openai and self.config.llm.openai.models:
            for model in self.config.llm.openai.models:
                if model.name == model_name:
                    return self.providers.get('openai')
        
        if self.config.llm.ollama and self.config.llm.ollama.models:
            for model in self.config.llm.ollama.models:
                if model.name == model_name:
                    return self.providers.get('ollama')
        
        # Default to OpenAI if available
        if 'openai' in self.providers:
            return self.providers['openai']
        
        # Fallback to Ollama
        if 'ollama' in self.providers:
            return self.providers['ollama']
        
        return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        use_fallback: bool = True,
        **kwargs
    ) -> ModelResponse:
        """Generate response with automatic fallback."""
        
        model_name = model_name or self.config.llm.default_model
        provider = self.get_provider(model_name)
        
        if not provider:
            raise RuntimeError(f"No provider available for model: {model_name}")
        
        try:
            return provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            print(f"[ModelManager] Primary provider failed: {str(e)}")
            
            if not use_fallback or not self.config.cost.fallback_to_local:
                raise
            
            # Try fallback to local model
            print("[ModelManager] Attempting fallback to local model...")
            if 'ollama' in self.providers:
                try:
                    ollama_provider = self.providers['ollama']
                    return ollama_provider.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model_name="llama3",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                except Exception as fallback_error:
                    print(f"[ModelManager] Fallback also failed: {str(fallback_error)}")
                    raise RuntimeError(f"Both primary and fallback providers failed") from fallback_error
            
            raise
