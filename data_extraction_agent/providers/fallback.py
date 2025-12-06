"""Fallback Chain - Automatic retry with alternative providers.

Implements provider-hopping fallback for resilient extraction.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from data_extraction_agent.providers.factory import (
    Provider,
    ProviderFactory,
)

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result from a fallback chain invocation."""

    success: bool
    content: str
    provider: Optional[str] = None
    model_id: Optional[str] = None
    attempts: int = 0
    total_time_ms: float = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# Default fallback chain: Free/cheap providers first
DEFAULT_FALLBACK_CHAIN = [
    (Provider.GROQ, "llama-3.1-8b-instant"),  # Free, fastest
    (Provider.GROQ, "llama-3.3-70b-versatile"),  # Free, better quality
    (Provider.OPENROUTER, "google/gemini-2.0-flash-exp:free"),  # Free
    (Provider.TOGETHER, "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),  # ~$0.05/M
    (Provider.GOOGLE, "gemini-2.0-flash-exp"),  # Low cost
    (Provider.ANTHROPIC, "claude-3-5-haiku-latest"),  # Cheap commercial
    (Provider.ANTHROPIC, "claude-sonnet-4-20250514"),  # Last resort
]

# Budget chain: Only free/very cheap options
BUDGET_FALLBACK_CHAIN = [
    (Provider.GROQ, "llama-3.1-8b-instant"),
    (Provider.GROQ, "llama-3.3-70b-versatile"),
    (Provider.OPENROUTER, "google/gemini-2.0-flash-exp:free"),
    (Provider.HUGGINGFACE, "Qwen/Qwen2.5-7B-Instruct"),
]

# Quality chain: Best models regardless of cost
QUALITY_FALLBACK_CHAIN = [
    (Provider.ANTHROPIC, "claude-sonnet-4-20250514"),
    (Provider.GOOGLE, "gemini-1.5-pro"),
    (Provider.TOGETHER, "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    (Provider.GROQ, "qwen-qwq-32b"),
]


class FallbackChain:
    """Chain that automatically falls back to alternative providers on failure."""

    def __init__(
        self,
        provider_chain: Optional[list[tuple[Provider, str]]] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
    ):
        """Initialize the fallback chain.

        Args:
            provider_chain: List of (provider, model_id) tuples to try
            max_retries: Max retries per provider before moving to next
            retry_delay: Delay between retries in seconds
            timeout: Timeout per request in seconds
        """
        self.provider_chain = provider_chain or DEFAULT_FALLBACK_CHAIN
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._models_cache: dict[str, BaseChatModel] = {}

    @classmethod
    def budget(cls) -> "FallbackChain":
        """Create a budget-optimized fallback chain."""
        return cls(provider_chain=BUDGET_FALLBACK_CHAIN)

    @classmethod
    def quality(cls) -> "FallbackChain":
        """Create a quality-optimized fallback chain."""
        return cls(provider_chain=QUALITY_FALLBACK_CHAIN)

    def _get_model(self, provider: Provider, model_id: str) -> Optional[BaseChatModel]:
        """Get or create a model instance."""
        cache_key = f"{provider.value}:{model_id}"
        if cache_key not in self._models_cache:
            if not ProviderFactory.is_available(provider):
                return None
            try:
                self._models_cache[cache_key] = ProviderFactory.create(
                    provider, model_id=model_id
                )
            except Exception as e:
                logger.warning(f"Failed to create {cache_key}: {e}")
                return None
        return self._models_cache[cache_key]

    def invoke(
        self,
        prompt: Union[str, list[BaseMessage]],
        **kwargs: Any,
    ) -> FallbackResult:
        """Invoke the chain with automatic fallback.

        Args:
            prompt: The prompt to send (string or messages)
            **kwargs: Additional arguments passed to the model

        Returns:
            FallbackResult with the response or error information
        """
        start_time = time.time()
        errors = []
        attempts = 0

        # Convert string prompt to message
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt

        # Try each provider in the chain
        for provider, model_id in self.provider_chain:
            model = self._get_model(provider, model_id)
            if model is None:
                continue

            # Try with retries
            for retry in range(self.max_retries):
                attempts += 1
                try:
                    response = model.invoke(messages, **kwargs)
                    content = response.content if hasattr(response, "content") else str(response)

                    return FallbackResult(
                        success=True,
                        content=content,
                        provider=provider.value,
                        model_id=model_id,
                        attempts=attempts,
                        total_time_ms=(time.time() - start_time) * 1000,
                        errors=errors,
                    )

                except Exception as e:
                    error_msg = f"{provider.value}/{model_id} attempt {retry + 1}: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

                    if retry < self.max_retries - 1:
                        time.sleep(self.retry_delay)

        # All providers failed
        return FallbackResult(
            success=False,
            content="All providers failed. Check API keys and network connectivity.",
            attempts=attempts,
            total_time_ms=(time.time() - start_time) * 1000,
            errors=errors,
        )

    async def ainvoke(
        self,
        prompt: Union[str, list[BaseMessage]],
        **kwargs: Any,
    ) -> FallbackResult:
        """Async version of invoke."""
        import asyncio

        start_time = time.time()
        errors = []
        attempts = 0

        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt

        for provider, model_id in self.provider_chain:
            model = self._get_model(provider, model_id)
            if model is None:
                continue

            for retry in range(self.max_retries):
                attempts += 1
                try:
                    response = await model.ainvoke(messages, **kwargs)
                    content = response.content if hasattr(response, "content") else str(response)

                    return FallbackResult(
                        success=True,
                        content=content,
                        provider=provider.value,
                        model_id=model_id,
                        attempts=attempts,
                        total_time_ms=(time.time() - start_time) * 1000,
                        errors=errors,
                    )

                except Exception as e:
                    error_msg = f"{provider.value}/{model_id} attempt {retry + 1}: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

                    if retry < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)

        return FallbackResult(
            success=False,
            content="All providers failed.",
            attempts=attempts,
            total_time_ms=(time.time() - start_time) * 1000,
            errors=errors,
        )

    def get_first_available_model(self) -> Optional[BaseChatModel]:
        """Get the first available model in the chain."""
        for provider, model_id in self.provider_chain:
            model = self._get_model(provider, model_id)
            if model is not None:
                return model
        return None
