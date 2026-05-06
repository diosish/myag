"""
Configuration module for AutoDevAgent.
Uses pydantic-settings for validation and environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""
    name: str
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    models: List[ModelConfig] = []


class OllamaConfig(BaseModel):
    """Ollama local model configuration."""
    base_url: str = "http://localhost:11434"
    models: List[ModelConfig] = []


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    default_model: str = "gpt-4"
    openai: Optional[OpenAIConfig] = None
    ollama: Optional[OllamaConfig] = None


class CostConfig(BaseModel):
    """Cost management configuration."""
    max_cost_per_task: float = 5.0
    max_daily_cost: float = 20.0
    routing_strategy: Literal["budget", "balanced", "premium"] = "balanced"
    fallback_to_local: bool = True


class TokenConfig(BaseModel):
    """Token limit configuration."""
    max_tokens_per_request: int = 4096
    context_reservation: int = 1000
    warning_threshold: float = 0.8


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    directory: str = "./data/cache"
    semantic_search: bool = False
    ttl: int = 86400
    max_size_mb: int = 1000


class GitConfig(BaseModel):
    """Git integration configuration."""
    enabled: bool = True
    auto_commit: bool = False
    commit_prefix: str = "[AutoDev]"


class ObsidianConfig(BaseModel):
    """Obsidian graph configuration."""
    enabled: bool = True
    graph_directory: str = "./data/graph"
    decisions_directory: str = "./data/decisions"
    auto_link: bool = True


class SyncConfig(BaseModel):
    """Project sync configuration."""
    project_root: str = "."
    git: GitConfig = GitConfig()
    obsidian: ObsidianConfig = ObsidianConfig()


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""
    max_retries: int = 3
    retry_delay: int = 5
    parallel: bool = False
    default_complexity: int = Field(ge=1, le=5, default=3)
    mode: Literal["fast-track", "standard", "enterprise"] = "standard"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: str = "./logs/adev.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_sensitive: bool = False
    mask_secrets: bool = True


def expand_env_variables(value: str) -> str:
    """Expand environment variables in format ${VAR_NAME} or $VAR_NAME."""
    if not isinstance(value, str):
        return value
    
    # Match ${VAR_NAME:-default} pattern (with default value)
    pattern_with_default = r'\$\{([^}:]+):-([^}]*)\}'
    
    def replace_with_default(match):
        env_var = match.group(1)
        default_value = match.group(2)
        return os.environ.get(env_var, default_value)
    
    result = re.sub(pattern_with_default, replace_with_default, value)
    
    # Match ${VAR_NAME} pattern (without default)
    pattern = r'\$\{([^}]+)\}'
    
    def replace(match):
        env_var = match.group(1)
        return os.environ.get(env_var, match.group(0))
    
    result = re.sub(pattern, replace, result)
    
    # Also match $VAR_NAME (without braces)
    pattern_simple = r'\$([A-Za-z_][A-Za-z0-9_]*)'
    result = re.sub(pattern_simple, lambda m: os.environ.get(m.group(1), m.group(0)), result)
    
    return result


class Config(BaseSettings):
    """Main application configuration."""
    
    llm: LLMConfig = LLMConfig()
    cost: CostConfig = CostConfig()
    tokens: TokenConfig = TokenConfig()
    cache: CacheConfig = CacheConfig()
    sync: SyncConfig = SyncConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    logging: LoggingConfig = LoggingConfig()
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    @validator('llm', pre=True)
    def validate_llm(cls, v):
        if isinstance(v, dict):
            # Expand environment variables in api_key
            if 'openai' in v and isinstance(v['openai'], dict):
                if 'api_key' in v['openai']:
                    v['openai']['api_key'] = expand_env_variables(v['openai']['api_key'])
        return v
    
    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        # Check OpenAI models
        if self.llm.openai:
            for model in self.llm.openai.models:
                if model.name == model_name:
                    return model
        
        # Check Ollama models
        if self.llm.ollama:
            for model in self.llm.ollama.models:
                if model.name == model_name:
                    return model
        
        return None
    
    def resolve_path(self, path: str) -> Path:
        """Resolve a config path relative to project root."""
        resolved = expand_env_variables(path)
        if os.path.isabs(resolved):
            return Path(resolved)
        return Path(self.sync.project_root) / resolved


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return Config()
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Expand environment variables in the raw config
    def expand_dict(d):
        if isinstance(d, dict):
            return {k: expand_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [expand_dict(item) for item in d]
        elif isinstance(d, str):
            return expand_env_variables(d)
        return d
    
    expanded_config = expand_dict(raw_config)
    
    return Config(**expanded_config)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config()
    return _config
