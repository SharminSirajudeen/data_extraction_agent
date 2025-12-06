"""Fallback Chain - Automatic retry with alternative providers.

Uses the provider registry for dynamic provider selection.
No hardcoded provider names in the fallback logic.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from data_extraction_agent.providers.base import ProviderRegistry, ProviderTier

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
    errors: list[str] = field(default_factory=list)


class FallbackChain:
    """Chain that automatically falls back to alternative providers on failure.

    Uses the provider registry - no hardcoded provider names.
    Dynamically builds fallback chain based on available providers.
    """

    def __init__(
        self,
        provider_names: Optional[list[str]] = None,
        model_type: str = "default",
        max_retries: int = 2,
        retry_delay: float = 1.0,
        prefer_free: bool = True,
        prefer_speed: bool = False,
    ):
        """Initialize the fallback chain.

        Args:
            provider_names: Explicit list of provider names to try (optional)
            model_type: Model type to use (default, fast, balanced, etc.)
            max_retries: Max retries per provider before moving to next
            retry_delay: Delay between retries in seconds
            prefer_free: Prioritize free tier providers
            prefer_speed: Prioritize faster providers
        """
        self.provider_names = provider_names
        self.model_type = model_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prefer_free = prefer_free
        self.prefer_speed = prefer_speed
        self._models_cache: dict[str, BaseChatModel] = {}

    @classmethod
    def budget(cls) -> "FallbackChain":
        """Create a budget-optimized fallback chain (free tier only)."""
        free_providers = [
            c.name for c in ProviderRegistry.get_by_tier(ProviderTier.FREE)
        ]
        return cls(provider_names=free_providers, prefer_free=True)

    @classmethod
    def fast(cls) -> "FallbackChain":
        """Create a speed-optimized fallback chain."""
        return cls(prefer_speed=True, prefer_free=False)

    @classmethod
    def quality(cls) -> "FallbackChain":
        """Create a quality-optimized fallback chain (premium first)."""
        # Premium -> Standard -> Budget -> Free
        providers = []
        for tier in [ProviderTier.PREMIUM, ProviderTier.STANDARD, ProviderTier.BUDGET, ProviderTier.FREE]:
            providers.extend([c.name for c in ProviderRegistry.get_by_tier(tier)])
        return cls(provider_names=providers, prefer_free=False)

    def _get_provider_chain(self) -> list[str]:
        """Get ordered list of providers to try."""
        if self.provider_names:
            # Use explicit list, filtered to available only
            return [
                name for name in self.provider_names
                if ProviderRegistry.get(name) and ProviderRegistry.get(name).is_available()
            ]

        # Dynamic ordering based on preferences
        available = ProviderRegistry.get_available()

        if self.prefer_free:
            available.sort(
                key=lambda p: (
                    0 if p.tier == ProviderTier.FREE else 1,
                    -p.speed_tokens_per_sec if self.prefer_speed else 0,
                )
            )
        elif self.prefer_speed:
            available.sort(key=lambda p: -p.speed_tokens_per_sec)

        return [p.name for p in available]

    def _get_model(self, provider_name: str) -> Optional[BaseChatModel]:
        """Get or create a model instance."""
        cache_key = f"{provider_name}:{self.model_type}"
        if cache_key not in self._models_cache:
            try:
                self._models_cache[cache_key] = ProviderRegistry.create_client(
                    provider_name=provider_name,
                    model_type=self.model_type,
                )
            except Exception as e:
                logger.warning(f"Failed to create {provider_name} client: {e}")
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

        # Get provider chain dynamically
        provider_chain = self._get_provider_chain()

        if not provider_chain:
            return FallbackResult(
                success=False,
                content="No providers available. Configure at least one API key.",
                errors=["No providers configured"],
            )

        # Try each provider
        for provider_name in provider_chain:
            model = self._get_model(provider_name)
            if model is None:
                continue

            config = ProviderRegistry.get(provider_name)
            model_id = config.get_model_id(self.model_type) if config else "unknown"

            # Try with retries
            for retry in range(self.max_retries):
                attempts += 1
                try:
                    response = model.invoke(messages, **kwargs)
                    content = response.content if hasattr(response, "content") else str(response)

                    return FallbackResult(
                        success=True,
                        content=content,
                        provider=provider_name,
                        model_id=model_id,
                        attempts=attempts,
                        total_time_ms=(time.time() - start_time) * 1000,
                        errors=errors,
                    )

                except Exception as e:
                    error_msg = f"{provider_name} attempt {retry + 1}: {str(e)}"
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

        provider_chain = self._get_provider_chain()

        if not provider_chain:
            return FallbackResult(
                success=False,
                content="No providers available.",
                errors=["No providers configured"],
            )

        for provider_name in provider_chain:
            model = self._get_model(provider_name)
            if model is None:
                continue

            config = ProviderRegistry.get(provider_name)
            model_id = config.get_model_id(self.model_type) if config else "unknown"

            for retry in range(self.max_retries):
                attempts += 1
                try:
                    response = await model.ainvoke(messages, **kwargs)
                    content = response.content if hasattr(response, "content") else str(response)

                    return FallbackResult(
                        success=True,
                        content=content,
                        provider=provider_name,
                        model_id=model_id,
                        attempts=attempts,
                        total_time_ms=(time.time() - start_time) * 1000,
                        errors=errors,
                    )

                except Exception as e:
                    error_msg = f"{provider_name} attempt {retry + 1}: {str(e)}"
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
        for provider_name in self._get_provider_chain():
            model = self._get_model(provider_name)
            if model is not None:
                return model
        return None
