from __future__ import annotations
from typing import Optional, Dict, Any, List, TypeVar, Type, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import yaml
import time
import logging
from contextlib import contextmanager

from pydantic import BaseModel, ValidationError

from .prompts import PromptManager
from .providers.base import ChatRequest, ModelResponse, ModelError, ModelTimeout
from .providers.ollama import OllamaProvider
from .providers.openai_sdk import OpenAIProvider
from .providers.groq_provider import GroqProvider
from .services.marker import MarkerService

logger = logging.getLogger(__name__)


class Provider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class Service(Enum):
    MARKER = "marker" #special case document processing service

@dataclass(frozen=True)
class TaskConfig:
    provider: str
    model: str
    params: Dict[str, Any]
    prompt_ref: Optional[str] #e.g. "vision/analyze@v1"


class ModelManager:
    def __init__(self, config_path: Union[Path, str], prompts_dir: Optional[Path] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._providers = {}
        self._stats = {} #performance tracking

        #initialize prompt manager
        if prompts_dir:
            self.prompts = PromptManager(prompts_dir)
        else:
            src_root = Path(__file__).parents[1]
            self.prompts = PromptManager(src_root.parent / "prompts")
        
        self._marker = None #lazy load marker service

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        if 'providers' not in config:
            raise ValueError("Config missing 'providers'")
        if 'tasks' not in config:
            raise ValueError("Config missing 'tasks'")
        
        for task_name, task_cfg in config['tasks'].items():
            if 'provider' not in task_cfg:
                raise ValueError(f"Task '{task_name}' missing provider")
            if 'model' not in task_cfg:
                raise ValueError(f"Task '{task_name}' missing model")
            
            provider_name = task_cfg['provider']
            if provider_name not in config['providers'] and provider_name not in config.get('services', {}):
                raise ValueError(f"Task '{task_name}' references unknown provider '{provider_name}'")
        
        return config

    @property
    def marker(self):
        """Access Marker service directly"""
        if self._marker is None:
            # Safely extract marker settings, handling None values at each level
            services = self.config.get('services') if self.config else None
            marker_config = services.get('marker') if services else None
            settings = marker_config.get('settings') if marker_config else None
            
            # Ensure settings is a dictionary for **kwargs unpacking
            if settings is None:
                settings = {}
            elif not isinstance(settings, dict):
                settings = {}
                
            self._marker = MarkerService(**settings)
        return self._marker

    def _get_provider(self, provider_name: str):
        if provider_name in self._providers:
            return self._providers[provider_name]
        if provider_name not in self.config['providers']:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_cfg = self.config["providers"][provider_name]
        provider_type = provider_cfg["type"]
        settings = provider_cfg.get("settings", {})

        if provider_type == "ollama":
            provider = OllamaProvider(**settings)
        elif provider_type == "openai":
            provider = OpenAIProvider(**settings)
        elif provider_type == "groq":
            provider = GroqProvider(**settings)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        self._providers[provider_name] = provider
        logger.info(f"initialized provider: {provider_name}")
        return provider

    def call(self, task: str, prompt_ref: str, variables: Dict[str, Any], schema: Optional[Type[BaseModel]] = None, images: Optional[List[bytes]] = None, messages_override: Optional[List[Dict[str, str]]] = None, **params_override) -> ModelResponse[BaseModel]:
        start_time = time.perf_counter()
        
        if task not in self.config["tasks"]:
            raise ValueError(f"Unknown task: {task}")
        
        task_cfg = self.config["tasks"][task]
        provider_name = task_cfg["provider"]
        model_name = task_cfg["model"]

        if messages_override:
            # If an override is provided, use it directly. This is for special cases like code repair.
            rendered = messages_override
            print(f"Using message override for task '{task}'. Bypassing prompt template.")
        else:
            # Otherwise, render the prompt as usual from a file.
            rendered = self.prompts.render(prompt_ref, variables)

        params = {**task_cfg.get("params", {}), **params_override}
        
        task_timeout = task_cfg.get("timeout")
        if task_timeout:
            # Ensure custom timeout in params_override takes precedence
            params.setdefault("timeout", task_timeout)
        
        request = ChatRequest(
            model=model_name,
            messages=rendered,
            images=images,
            params=params,
            schema=schema,
            extra_body=task_cfg.get("extra_body") or None,
        )
        
        provider = self._get_provider(provider_name)
        try:
            response = provider.chat(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._track_stats(task, elapsed_ms, success=True)
            return response
        except (ModelTimeout, ModelError) as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._track_stats(task, elapsed_ms, success=False)
            raise

    def _track_stats(self, task: str, latency_ms: float, success: bool):
        if task not in self._stats:
            self._stats[task] = {
                'total_calls': 0,
                'successful_calls': 0,
                'total_latency_ms': 0
            }

        stats = self._stats[task]
        stats['total_calls'] += 1
        if success:
            stats['successful_calls'] += 1
            stats['total_latency_ms'] += latency_ms
    
    def get_stats(self, task: Optional[str] = None) -> Dict:
        if task:
            return self._stats.get(task, {})
        return self._stats
    
    def cleanup(self):
        for name, provider in self._providers.items():
            if hasattr(provider, 'cleanup'):
                try:
                    provider.cleanup()
                    logger.info(f"Cleaned up provider: {name}")
                except Exception as e:
                    logger.error(f"Cleanup failed for {name}: {e}")
        
        self._providers.clear()
    
    @contextmanager
    def session(self):
        try:
            yield self
        finally:
            self.cleanup()